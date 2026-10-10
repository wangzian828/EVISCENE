"""Evaluator-private reference-chain recovery metrics.

The scorer is deliberately narrower than open-ended world reconstruction. It asks
whether a runtime trace recovered the decisive fact/evidence relationships frozen by
the data pipeline. Unrelated investigation material is diagnostic only unless it is
placed in the final verdict basis.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from eviscene.orchestrator.llm_backend import LLMBackend
from eviscene.orchestrator.evidence_semantics import (
    evidence_is_qualified_for_stance,
)
from eviscene.trajectory.scoring import (
    _basis_same_capture_source_context,
    _canonical_url,
    _evidence_matches_reference,
    _family_key,
    _has_actual_visual_bridge,
    _mapping,
    _match_gold_facts,
    _rows,
    _successful_call_ids,
    _token_f1,
    _tokens,
    _valid_findings,
)


REFERENCE_CHAIN_SCHEMA_VERSION = "ifv-reference-chain-metrics-v2"
SEMANTIC_TEXT_MATCH_THRESHOLD = 0.72
DEFAULT_MAX_LLM_CANDIDATES_PER_FACT = 3


@dataclass(frozen=True)
class SemanticMatchDecision:
    match: bool
    confidence: float
    reason: str
    reference_index: int | None = None


class ReferenceEvidenceMatcher(Protocol):
    """Constrained semantic matcher for an already selected gold fact."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return non-secret matcher configuration and usage metadata."""

    async def match(
        self,
        *,
        gold_fact: Mapping[str, Any],
        runtime_fact: Mapping[str, Any],
        evidence: Mapping[str, Any],
        references: Sequence[Mapping[str, Any]],
    ) -> SemanticMatchDecision:
        """Judge whether evidence recovers one of the supplied reference edges."""


def _extract_json_object(text: str) -> Mapping[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("semantic matcher did not return a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("semantic matcher JSON must be an object")
    return payload


class LLMReferenceEvidenceMatcher:
    """LLM judge restricted to matching one runtime edge to frozen references."""

    SYSTEM_PROMPT = """\
You are a benchmark evaluator matching one runtime Evidence object to a fixed,
evaluator-private reference chain. This is not open-ended fact checking.

Return match=true only when the runtime evidence directly recovers the same factual
relationship as at least one supplied acceptable reference and has the same support
or refute direction. Alternative official URLs, page versions, image asset URLs, and
wording are allowed. A shared topic, entity, logo, or generic background page is not
enough. A direct same-capture comparison may recover a source-page caption when both
refer to the same accepted source family and depicted capture.

Do not invent a new target or reward extra facts. Return exactly one JSON object:
{"match":bool,"confidence":number,"reference_index":integer|null,"reason":string}
"""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        provider: str,
        model: str,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "confidence_threshold": self.confidence_threshold,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    async def match(
        self,
        *,
        gold_fact: Mapping[str, Any],
        runtime_fact: Mapping[str, Any],
        evidence: Mapping[str, Any],
        references: Sequence[Mapping[str, Any]],
    ) -> SemanticMatchDecision:
        payload = {
            "gold_fact": {
                "kind": gold_fact.get("kind"),
                "statement": gold_fact.get("statement"),
                "expected_status": gold_fact.get("expected_status"),
                "visual_anchor": gold_fact.get("visual_anchor"),
            },
            "runtime_fact": {
                "kind": runtime_fact.get("kind"),
                "statement": runtime_fact.get("statement"),
                "status": runtime_fact.get("status"),
            },
            "runtime_evidence": {
                key: evidence.get(key)
                for key in (
                    "evidence_kind",
                    "source_url",
                    "source_family",
                    "source_class",
                    "exact_text",
                    "stance",
                    "quality",
                    "directness",
                    "claim_binding",
                    "relation_scope",
                    "relation_stance",
                    "same_subject_or_scene",
                    "same_capture_or_near_duplicate",
                    "likely_different_original_capture",
                    "edit_evidence_present",
                    "risk_flags",
                )
            },
            "acceptable_references": [
                {
                    key: reference.get(key)
                    for key in (
                        "source_id",
                        "canonical_url",
                        "exact_span",
                        "stance",
                        "role",
                        "directness",
                        "source_family",
                    )
                }
                for reference in references
            ],
        }
        response = await self.backend.get_response(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            max_tokens=500,
        )
        self.calls += 1
        self.prompt_tokens += int(response.prompt_tokens or 0)
        self.completion_tokens += int(response.completion_tokens or 0)
        parsed = _extract_json_object(response.text)
        confidence = float(parsed.get("confidence", 0.0))
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        raw_index = parsed.get("reference_index")
        reference_index = (
            int(raw_index)
            if isinstance(raw_index, int)
            and not isinstance(raw_index, bool)
            and 0 <= raw_index < len(references)
            else None
        )
        matched = (
            parsed.get("match") is True
            and confidence >= self.confidence_threshold
        )
        if matched and reference_index is None and len(references) == 1:
            reference_index = 0
        if matched and reference_index is None:
            matched = False
        return SemanticMatchDecision(
            match=matched,
            confidence=confidence,
            reference_index=reference_index,
            reason=str(parsed.get("reason", ""))[:1000],
        )


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _stance_matches(
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    return str(evidence.get("stance", "")) == str(
        reference.get("stance", "")
    )


def _qualified_semantic_candidate(
    evidence: Mapping[str, Any],
    successful_calls: set[str],
) -> bool:
    stance = str(evidence.get("stance", "")).strip()
    return bool(
        stance in {"support", "refute"}
        and evidence_is_qualified_for_stance(evidence, stance)
        and str(evidence.get("function_call_id", "")).strip()
        in successful_calls
    )


def _semantic_stance_by_evidence(
    investigation: Mapping[str, Any],
    *,
    valid_finding_ids: set[str],
) -> Dict[str, str]:
    """Recover accepted proposition-relative direction without mutating Evidence."""

    resolved: Dict[str, str] = {}
    for decision in _rows(investigation.get("evidence_decisions")):
        output = _mapping(decision.get("output"))
        assessment = str(output.get("assessment", "")).strip()
        if assessment not in {"supported", "refuted"}:
            continue
        finding_ids = {
            str(item)
            for item in decision.get("finding_ids", []) or []
            if str(item)
        }
        if finding_ids and not finding_ids & valid_finding_ids:
            continue
        stance = "support" if assessment == "supported" else "refute"
        for evidence_id in output.get("selected_evidence_ids", []) or []:
            rendered = str(evidence_id).strip()
            if rendered:
                resolved[rendered] = stance
    return resolved


def _semantic_evidence_view(
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    semantic_stance_by_evidence: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Create scorer-only Evidence rows with accepted semantic direction."""

    result: Dict[str, Dict[str, Any]] = {}
    for evidence_id, evidence in evidence_by_id.items():
        row = dict(evidence)
        semantic_stance = semantic_stance_by_evidence.get(evidence_id)
        if semantic_stance:
            row["stance"] = semantic_stance
        result[evidence_id] = row
    return result


def _deterministic_semantic_match(
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[str, float] | None:
    """Accept only high-confidence snapshot/page variants without an LLM."""

    if not _stance_matches(evidence, reference):
        return None
    evidence_family = _family_key(evidence.get("source_family"))
    reference_family = _family_key(reference.get("source_family"))
    same_family = bool(
        evidence_family
        and reference_family
        and evidence_family == reference_family
    )
    if not same_family:
        return None

    evidence_text = _normalized_text(evidence.get("exact_text"))
    reference_text = _normalized_text(reference.get("exact_span"))
    text_similarity = _token_f1(evidence_text, reference_text)
    reference_tokens = _tokens(reference_text)
    evidence_tokens = _tokens(evidence_text)
    reference_token_recall = (
        len(reference_tokens & evidence_tokens) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    if evidence_text and reference_text and (
        evidence_text in reference_text
        or reference_text in evidence_text
        or text_similarity >= SEMANTIC_TEXT_MATCH_THRESHOLD
        or reference_token_recall >= 0.85
    ):
        return "same_source_text_equivalent", max(
            SEMANTIC_TEXT_MATCH_THRESHOLD,
            text_similarity,
            reference_token_recall,
        )

    if (
        str(evidence.get("evidence_kind", "")) == "reference_comparison"
        and str(evidence.get("claim_binding", "")) == "same_capture"
        and bool(evidence.get("same_capture_or_near_duplicate", True))
        and not bool(evidence.get("likely_different_original_capture", False))
    ):
        return "same_source_same_capture", float(
            evidence.get("confidence", 0.9) or 0.9
        )

    if (
        _canonical_url(evidence.get("source_url"))
        == _canonical_url(reference.get("canonical_url"))
        and text_similarity >= 0.45
    ):
        return "same_url_semantic_span", text_similarity
    return None


def _match_record(
    *,
    evidence_id: str,
    reference: Mapping[str, Any],
    method: str,
    confidence: float,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "reference_source_id": str(reference.get("source_id", "")),
        "method": method,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "reason": reason,
    }


async def score_reference_chain_trace(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    semantic_matcher: ReferenceEvidenceMatcher | None = None,
    score_metadata: Mapping[str, Any] | None = None,
    max_llm_candidates_per_fact: int = DEFAULT_MAX_LLM_CANDIDATES_PER_FACT,
) -> Dict[str, Any]:
    """Score one canonical trace against the frozen decisive reference chain."""

    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    facts = _rows(investigation.get("facts"))
    decisive_ids = {
        str(item)
        for item in investigation.get("decisive_fact_ids", []) or []
    }
    decisive_facts = [
        fact for fact in facts if str(fact.get("fact_id", "")) in decisive_ids
    ]
    runtime_fact_by_id = {
        str(item.get("fact_id", "")): item for item in decisive_facts
    }
    tasks = {
        str(item.get("task_id", "")): item
        for item in _rows(investigation.get("tasks"))
        if str(item.get("task_id", ""))
    }
    evidence_by_id = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    findings = _rows(investigation.get("findings"))
    steps = _rows(state.get("all_steps"))
    successful_calls = _successful_call_ids(steps)
    valid_finding_ids = _valid_findings(
        findings,
        tasks,
        evidence_by_id,
        successful_calls,
    )
    semantic_evidence_by_id = _semantic_evidence_view(
        evidence_by_id,
        semantic_stance_by_evidence=_semantic_stance_by_evidence(
            investigation,
            valid_finding_ids=valid_finding_ids,
        ),
    )
    basis = _mapping(
        trace.get("verdict_basis") or investigation.get("verdict_basis")
    )
    recovered_source_context_by_fact = _basis_same_capture_source_context(
        investigation,
        basis,
    )
    gold_facts = _rows(gold.get("decisive_facts"))
    fact_matches = _match_gold_facts(
        decisive_facts,
        gold_facts,
        recovered_source_context_by_fact=recovered_source_context_by_fact,
    )
    fact_match_by_gold = {
        str(item["gold_fact_id"]): item for item in fact_matches
    }
    basis_evidence_ids = {
        str(item) for item in basis.get("evidence_ids", []) or []
    }

    reference_fact_rows: List[Dict[str, Any]] = []
    recovered_evidence_ids: set[str] = set()
    matcher_errors: List[Dict[str, Any]] = []
    llm_candidate_count = 0

    for gold_fact in gold_facts:
        gold_fact_id = str(gold_fact.get("fact_id", ""))
        fact_match = fact_match_by_gold.get(gold_fact_id, {})
        runtime_fact_id = str(fact_match.get("runtime_fact_id") or "")
        runtime_fact = runtime_fact_by_id.get(runtime_fact_id, {})
        references = _rows(gold_fact.get("acceptable_evidence"))
        related_findings = [
            finding
            for finding in findings
            if runtime_fact_id
            and runtime_fact_id
            in {str(item) for item in finding.get("fact_ids", []) or []}
            and str(finding.get("finding_id", "")) in valid_finding_ids
        ]
        related_evidence = list(
            {
                str(evidence_id): semantic_evidence_by_id[str(evidence_id)]
                for finding in related_findings
                for evidence_id in finding.get("evidence_ids", []) or []
                if str(evidence_id) in semantic_evidence_by_id
            }.values()
        )

        match_records: List[Dict[str, Any]] = []
        fact_recovered_evidence_ids: set[str] = set()
        for evidence in related_evidence:
            evidence_id = str(evidence.get("evidence_id", ""))
            exact_reference = next(
                (
                    reference
                    for reference in references
                    if _evidence_matches_reference(evidence, reference)
                ),
                None,
            )
            if exact_reference is not None:
                fact_recovered_evidence_ids.add(evidence_id)
                match_records.append(
                    _match_record(
                        evidence_id=evidence_id,
                        reference=exact_reference,
                        method="canonical_reference",
                        confidence=1.0,
                    )
                )
                continue
            if not _qualified_semantic_candidate(
                evidence,
                successful_calls,
            ):
                continue
            for reference in references:
                deterministic = _deterministic_semantic_match(
                    evidence,
                    reference,
                )
                if deterministic is None:
                    continue
                method, confidence = deterministic
                fact_recovered_evidence_ids.add(evidence_id)
                match_records.append(
                    _match_record(
                        evidence_id=evidence_id,
                        reference=reference,
                        method=method,
                        confidence=confidence,
                    )
                )
                break

        # A selected same-capture edge and direct assertion can complement each other
        # only within the same recovered source family and factual direction.
        recovered_rows = [
            semantic_evidence_by_id[evidence_id]
            for evidence_id in fact_recovered_evidence_ids
            if evidence_id in semantic_evidence_by_id
        ]
        recovered_families = {
            _family_key(item.get("source_family"))
            for item in recovered_rows
            if _family_key(item.get("source_family"))
        }
        recovered_bindings = {
            str(item.get("claim_binding", ""))
            for item in recovered_rows
        }
        recovered_stances = {
            str(item.get("stance", ""))
            for item in recovered_rows
        }
        for evidence in related_evidence:
            evidence_id = str(evidence.get("evidence_id", ""))
            if (
                evidence_id in fact_recovered_evidence_ids
                or evidence_id not in basis_evidence_ids
                or not _qualified_semantic_candidate(
                    evidence,
                    successful_calls,
                )
            ):
                continue
            family = _family_key(evidence.get("source_family"))
            binding = str(evidence.get("claim_binding", ""))
            stance = str(evidence.get("stance", ""))
            complementary = (
                binding == "source_assertion"
                and "same_capture" in recovered_bindings
            ) or (
                binding == "same_capture"
                and "source_assertion" in recovered_bindings
            )
            if (
                family
                and family in recovered_families
                and stance in recovered_stances
                and complementary
            ):
                fact_recovered_evidence_ids.add(evidence_id)
                match_records.append(
                    {
                        "evidence_id": evidence_id,
                        "reference_source_id": "",
                        "method": "same_source_chain_complement",
                        "confidence": 1.0,
                        "reason": (
                            "Qualified same-source capture binding and factual "
                            "assertion form one complementary evidence chain."
                        ),
                    }
                )

        if semantic_matcher is not None:
            unresolved_candidates = [
                evidence
                for evidence in related_evidence
                if str(evidence.get("evidence_id", ""))
                not in fact_recovered_evidence_ids
                and _qualified_semantic_candidate(evidence, successful_calls)
                and any(
                    _stance_matches(evidence, reference)
                    for reference in references
                )
            ]
            unresolved_candidates.sort(
                key=lambda item: (
                    str(item.get("evidence_id", ""))
                    not in basis_evidence_ids,
                    str(item.get("quality", "")) != "strong",
                    _family_key(item.get("source_family"))
                    not in {
                        _family_key(reference.get("source_family"))
                        for reference in references
                    },
                    str(item.get("evidence_id", "")),
                )
            )
            basis_candidates = [
                item
                for item in unresolved_candidates
                if str(item.get("evidence_id", "")) in basis_evidence_ids
            ]
            non_basis_candidates = [
                item
                for item in unresolved_candidates
                if str(item.get("evidence_id", "")) not in basis_evidence_ids
            ]
            selected_candidates = list(basis_candidates)
            if not fact_recovered_evidence_ids:
                remaining = max(
                    0,
                    max_llm_candidates_per_fact - len(selected_candidates),
                )
                selected_candidates.extend(non_basis_candidates[:remaining])
            for evidence in selected_candidates:
                if (
                    fact_recovered_evidence_ids
                    and str(evidence.get("evidence_id", ""))
                    not in basis_evidence_ids
                ):
                    continue
                llm_candidate_count += 1
                compatible_references = [
                    reference
                    for reference in references
                    if _stance_matches(evidence, reference)
                ]
                evidence_id = str(evidence.get("evidence_id", ""))
                try:
                    decision = await semantic_matcher.match(
                        gold_fact=gold_fact,
                        runtime_fact=runtime_fact,
                        evidence=evidence,
                        references=compatible_references,
                    )
                except Exception as exc:
                    matcher_errors.append(
                        {
                            "gold_fact_id": gold_fact_id,
                            "evidence_id": evidence_id,
                            "error": f"{type(exc).__name__}: {exc}"[:1000],
                        }
                    )
                    continue
                if not decision.match or decision.reference_index is None:
                    continue
                reference = compatible_references[decision.reference_index]
                fact_recovered_evidence_ids.add(evidence_id)
                match_records.append(
                    _match_record(
                        evidence_id=evidence_id,
                        reference=reference,
                        method="llm_semantic",
                        confidence=decision.confidence,
                        reason=decision.reason,
                    )
                )

        recovered_evidence_ids.update(fact_recovered_evidence_ids)
        fact_recovered = bool(runtime_fact_id)
        status_match = bool(fact_match.get("status_match", False))
        visual_anchor_recovered = bool(
            fact_recovered
            and (
                not gold_fact.get("visual_anchor")
                or _has_actual_visual_bridge(runtime_fact, related_evidence)
            )
        )
        evidence_recovered = bool(fact_recovered_evidence_ids)
        reference_fact_rows.append(
            {
                "gold_fact_id": gold_fact_id,
                "runtime_fact_id": runtime_fact_id or None,
                "fact_similarity": float(fact_match.get("similarity", 0.0)),
                "expected_status": str(
                    gold_fact.get("expected_status", "")
                ),
                "runtime_status": (
                    str(runtime_fact.get("status", ""))
                    if runtime_fact_id
                    else None
                ),
                "status_match": status_match,
                "visual_anchor_recovered": visual_anchor_recovered,
                "recovered_evidence_ids": sorted(
                    fact_recovered_evidence_ids
                ),
                "evidence_matches": match_records,
                "chain_recovered": all(
                    (
                        fact_recovered,
                        status_match,
                        visual_anchor_recovered,
                        evidence_recovered,
                    )
                ),
            }
        )

    denominator = len(gold_facts)
    fact_recovery_recall = (
        sum(bool(item["runtime_fact_id"]) for item in reference_fact_rows)
        / denominator
        if denominator
        else 1.0
    )
    chain_recovery_recall = (
        sum(bool(item["chain_recovered"]) for item in reference_fact_rows)
        / denominator
        if denominator
        else 1.0
    )
    evidence_recovery_recall = (
        sum(
            bool(item["recovered_evidence_ids"])
            for item in reference_fact_rows
        )
        / denominator
        if denominator
        else 1.0
    )
    basis_reference_precision = (
        len(basis_evidence_ids & recovered_evidence_ids)
        / len(basis_evidence_ids)
        if basis_evidence_ids
        else (1.0 if not gold_facts else 0.0)
    )
    return {
        "schema_version": REFERENCE_CHAIN_SCHEMA_VERSION,
        "diagnostic_role": "canonical_reference_chain_recovery_only",
        "case_id": str(gold.get("case_id") or trace.get("image_id") or ""),
        "score_metadata": dict(score_metadata or {}),
        "metrics": {
            "fact_recovery_recall": round(fact_recovery_recall, 6),
            "chain_recovery_recall": round(chain_recovery_recall, 6),
            "evidence_recovery_recall": round(
                evidence_recovery_recall,
                6,
            ),
            "basis_reference_precision": round(
                basis_reference_precision,
                6,
            ),
        },
        "reference_facts": reference_fact_rows,
        "basis_evidence_ids": sorted(basis_evidence_ids),
        "off_reference_chain_basis_evidence_ids": sorted(
            basis_evidence_ids - recovered_evidence_ids
        ),
        "semantic_matcher": {
            "enabled": semantic_matcher is not None,
            "mode": (
                "deterministic_plus_llm"
                if semantic_matcher is not None
                else "deterministic"
            ),
            "candidate_count": llm_candidate_count,
            "errors": matcher_errors,
            **(
                dict(semantic_matcher.metadata)
                if semantic_matcher is not None
                else {}
            ),
        },
    }
