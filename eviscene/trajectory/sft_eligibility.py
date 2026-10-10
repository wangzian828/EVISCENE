"""Private-gold SFT eligibility audit for completed teacher trajectories.

The audit judges whether a complete ReAct trajectory reached the right binary
decision about the factual content expressed by the image and whether its
recorded observations and sources are useful for training. It does not require
the runtime to expose a preconstructed target or route graph.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

from pydantic import Field

from eviscene.orchestrator.react_runtime import REACT_RUNTIME_SCHEMA_VERSION
from eviscene.tools.vision_utils import controlled_image_to_data_url
from eviscene.trajectory.semantic_reward import (
    SemanticRewardJudge,
    _StrictModel,
    _mapping,
    _rows,
    canonical_json as canonical_json,
    sha256_json,
)


SFT_ELIGIBILITY_SCHEMA_VERSION = "ifv-sft-eligibility-v5"
SFT_ELIGIBILITY_INPUT_VERSION = "ifv-sft-eligibility-input-v12"
SFT_ELIGIBILITY_PROMPT_VERSION = "ifv-sft-private-image-fact-gate-v8"
SFT_ELIGIBILITY_GENERATION_VERSION = "low-thinking-4096-v7"
SFT_ELIGIBILITY_POSTPROCESS_VERSION = "image-fact-safety-gate-v7"


SFT_ELIGIBILITY_SYSTEM_PROMPT = (
    "You are a frozen post-rollout SFT eligibility auditor. Judge whether the "
    "completed teacher trajectory correctly determined the factual content expressed "
    "by the supplied image and grounded that decision in retained observations. The "
    "private target describes the image fact and the expected binary verdict. "
    "Do not require the teacher to reproduce the target wording, private-target "
    "wording, a specific runtime ID, URL, original image, source span, or registered "
    "decision path. First assess target_scope independently from correctness, "
    "evidence sufficiency, retrieval quality, and overclaiming. Use direct_target "
    "when the teacher's central investigation tests the target's decisive relation, "
    "including its important identity, time, place, attribution, or physical "
    "condition. Use decisive_subfact when the teacher investigates a different "
    "image-grounded sub-fact that logically establishes the target verdict. For "
    "example, a dated source image proving a picture was taken in a different "
    "country and year can decisively refute a displayed event attribution. Use "
    "related_but_incomplete when the teacher investigates the same picture, event, "
    "entity, or general scene authenticity but does not test the target's decisive "
    "condition. For example, checking whether a garden scene looks real is related "
    "but incomplete when the target is whether a particular planter is unsupported "
    "and floating. Use unrelated_fact only when the teacher's central investigation "
    "is a genuinely different image fact, event, entity, or claim. Do not call a "
    "trajectory unrelated_fact merely because it missed a key condition, reached a "
    "wrong verdict, used weak observations, or did not reproduce the target wording. "
    "A raw observation is one successful tool result identified by its observation_id. "
    "Select only supplied observation IDs for the decisive and supporting fields. "
    "Successful empty searches may record that a directed query found no matching "
    "trace, but an empty search alone cannot support fake. Tool errors, malformed "
    "results, and access failures are limitations and cannot be selected. Retrieval "
    "history describes what the teacher actually investigated, but is process context "
    "rather than evidence by itself. A lack of matching results may only supplement "
    "an image-grounded chain "
    "when the history targets a named, plausibly authoritative source or bounded "
    "collection; generic web search failure, topical relatedness, or absence of a "
    "found original never decides the verdict. If final_visual_audit is present, "
    "treat it as a structured VLM observation available to the terminal judgment, "
    "not as a replacement for the cited raw observation chain. "
    "Assess retrieval_quality as effective when the trajectory's retrieval is "
    "targeted and converted into relevant inspection or retained observations; mixed when its "
    "central route is useful despite some noise or corrected turns; poor when its "
    "central route is generic, repeatedly low-yield, premise-led, ignores useful "
    "candidates, or treats non-results as a conclusion. Mark major overclaiming "
    "when the cited material does not establish the conclusion. Do not search and "
    "do not create human-review work. The candidate also contains the ordered "
    "ReAct actions, their tool observations, the unverified search/image-candidate "
    "ledger, and rejected intermediate policy outputs. The action history is "
    "process context: it shows what the teacher actually did, but it does not "
    "promote search candidates or titles into Evidence. A single rejected attempt "
    "that is "
    "followed by a materially different, successful investigation or decision is a "
    "recoverable minor error. Repeated duplicate tool attempts, repeated terminal "
    "verdict proposals after the same stated deficiency, or a rejection that the "
    "trajectory never substantively repairs are degraded teacher conduct. Classify "
    "trajectory_conduct as clean when there is no material rejected turn; "
    "recovered_minor when any rejected turn is clearly repaired and the final path "
    "is suitable to imitate; degraded_repetition when the trajectory loops or "
    "repeats a blocked behavior; or unresolved when the final path still depends "
    "on an unresolved rejected behavior. Rejected turns are not themselves SFT "
    "targets, but their presence is relevant to whether the accepted trajectory "
    "would teach good behavior. The candidate may include a final reader-facing "
    "fact_check_report and runtime-owned evidence citations. Treat the report as "
    "a summary to audit against the supplied Evidence and basis, never as new "
    "Evidence; a report that invents a source, observation, or stronger fact is "
    "major overclaiming."
)


class SFTEligibilityJudgment(_StrictModel):
    target_scope: Literal[
        "direct_target",
        "decisive_subfact",
        "related_but_incomplete",
        "unrelated_fact",
        "unclear",
    ]
    decision_support: Literal[
        "supports_real",
        "supports_fake",
        "supporting_only",
        "insufficient",
        "unclear",
    ]
    retrieval_quality: Literal["effective", "mixed", "poor"]
    decisive_observation_ids: List[str] = Field(
        default_factory=list,
        max_length=40,
    )
    supporting_observation_ids: List[str] = Field(
        default_factory=list,
        max_length=40,
    )
    overclaiming: Literal["none", "minor", "major"]
    boundary_assessment: Literal[
        "respected",
        "minor_issue",
        "major_issue",
    ]
    trajectory_conduct: Literal[
        "clean",
        "recovered_minor",
        "degraded_repetition",
        "unresolved",
    ]
    # Confidence is diagnostic-only. Some providers emit a five-point score
    # despite the JSON schema; normalize it downstream instead of failing an
    # otherwise complete SFT eligibility batch.
    confidence: float
    explanation: str = Field(min_length=1, max_length=1600)


def _text(*values: Any, limit: int = 4000) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()[:limit]
    return ""


def _unique(values: Iterable[Any], *, limit: int = 40) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= limit:
            break
    return result


def _string_values(value: Any, *, limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        return []
    return _unique(
        [
            item
            for item in value
            if isinstance(item, str) and item.strip()
        ],
        limit=limit,
    )


def _target_ids(row: Mapping[str, Any]) -> List[str]:
    return _unique(
        (
            row.get("case_id"),
            row.get("candidate_id"),
            row.get("assignment_id"),
            row.get("source_item_id"),
        ),
        limit=8,
    )


def _expected_verdict(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("expected_verdict")
        or row.get("factual_status")
        or row.get("label")
        or ""
    ).strip().lower()
    if value in {"real", "fake"}:
        return value
    if value == "supported":
        return "real"
    if value == "refuted":
        return "fake"
    raise ValueError("private target must declare supported/refuted or real/fake")


def _chain_rows(binding: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    chain = binding.get("chain")
    return _rows(chain)


def _reference_facts(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    evidence = _mapping(row.get("evidence"))
    binding = _mapping(evidence.get("binding"))

    def add(
        *,
        statement: Any = "",
        evidence_text: Any = "",
        source_url: Any = "",
        role: Any = "",
        verified_value: Any = "",
    ) -> None:
        statement_text = _text(statement, evidence_text)
        evidence_text_value = _text(evidence_text, statement)
        if not statement_text and not evidence_text_value:
            return
        facts.append(
            {
                "statement": statement_text,
                "verified_value": _text(verified_value, limit=1200),
                "evidence_text": evidence_text_value,
                "source_url": _text(source_url, limit=2000),
                "role": _text(role, "source_evidence", limit=120),
            }
        )

    for item in _chain_rows(binding):
        inner = _mapping(item.get("binding")) or item
        add(
            statement=item.get("semantic_audit_reason")
            or inner.get("target_claim"),
            evidence_text=item.get("exact_span")
            or inner.get("source_evidence_span")
            or inner.get("contradiction_relation_identity"),
            source_url=inner.get("source_url") or binding.get("source_url"),
            role=item.get("role") or item.get("type") or "evidence_chain",
            verified_value=inner.get("verified_value"),
        )

    add(
        statement=binding.get("target_claim") or evidence.get("target_claim"),
        evidence_text=(
            binding.get("source_evidence_span")
            or binding.get("candidate_exact_span")
            or binding.get("primary_exact_span")
        ),
        source_url=binding.get("source_url") or row.get("source_url"),
        role=binding.get("match") or "source_evidence",
        verified_value=binding.get("verified_value"),
    )

    for span in _rows(evidence.get("used_exact_spans")):
        add(
            evidence_text=span.get("exact_span") or span.get("text"),
            source_url=row.get("source_url"),
            role=span.get("role") or "source_exact_span",
        )

    for span in _rows(row.get("source_exact_spans")):
        add(
            evidence_text=span.get("text") or span.get("exact_span"),
            source_url=row.get("source_url"),
            role=span.get("role") or "source_exact_span",
        )

    return facts[:40]


def build_sft_target(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the final data-pipeline row into one generic ImageFact target."""

    ids = _target_ids(row)
    if not ids:
        raise ValueError("private target lacks case_id/candidate_id/assignment_id")

    claim_atom = _mapping(row.get("claim_atom"))
    decisive = _text(
        row.get("decisive_visual_atom"),
        _mapping(row.get("automatic_qa")).get("target_visual_atom_observation"),
        limit=4000,
    )
    construction = _mapping(row.get("construction_spec"))
    visible_facts = _string_values(row.get("visible_scene_facts"))
    if not visible_facts:
        visible_facts = _string_values(construction.get("visible_scene_facts"))
    if not visible_facts:
        qa = _mapping(row.get("automatic_qa"))
        visible_facts = _unique(
            [qa.get("target_visual_atom_observation")],
            limit=4,
        )

    visible_anchors = _unique(
        [
            decisive,
            *[_text(item, limit=1200) for item in visible_facts],
        ],
        limit=12,
    )
    statement = _text(
        row.get("target_claim"),
        row.get("primary_claim"),
        _mapping(row.get("decisive_fact")).get("statement"),
        decisive,
        limit=4000,
    )
    image_fact = {
        "statement": statement,
        "visible_anchors": visible_anchors,
        "subject": _text(claim_atom.get("subject"), limit=1200),
        "event_or_context": _text(
            claim_atom.get("event_or_context"),
            row.get("event_identity"),
            limit=1600,
        ),
        "relation": _text(
            claim_atom.get("relation"),
            claim_atom.get("relation_slot"),
            row.get("relation_identity"),
            limit=1200,
        ),
        "depicted_value": _text(
            claim_atom.get("depicted_value"),
            limit=1200,
        ),
    }
    return {
        "schema_version": "ifv-sft-target-v2",
        "case_id": ids[0],
        "case_id_aliases": ids,
        "expected_verdict": _expected_verdict(row),
        "image_fact": image_fact,
        "reference_facts": _reference_facts(row),
    }


def _tool_result_mapping(step: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = step.get("tool_result")
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return _mapping(parsed)


_TRACE_MEDIA_KEYS = {
    "image_input",
    "image",
    "image_url",
    "data_url",
    "base64",
    "content_bytes",
    "raw_html",
    "html",
}


def _compact_trace_value(
    value: Any,
    *,
    depth: int = 0,
    max_string: int = 6000,
) -> Any:
    """Bound trace observations without rewriting their textual meaning."""

    if depth > 5:
        return "[nested content omitted]"
    if isinstance(value, str):
        text = value.strip()
        if len(text) <= max_string:
            return text
        return text[: max_string - 1].rstrip() + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, child in list(value.items())[:80]:
            key = str(raw_key)
            if key.casefold() in _TRACE_MEDIA_KEYS:
                continue
            result[key] = _compact_trace_value(
                child,
                depth=depth + 1,
                max_string=max_string,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _compact_trace_value(
                child,
                depth=depth + 1,
                max_string=max_string,
            )
            for child in list(value)[:40]
        ]
    return _compact_trace_value(
        str(value),
        depth=depth + 1,
        max_string=max_string,
    )


def _project_search_rows(value: Any) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in _rows(value)[:12]:
        item: Dict[str, Any] = {}
        for key in (
            "rank",
            "title",
            "snippet",
            "source",
            "url",
            "link",
            "image_url",
            "thumbnail",
            "imageUrl",
            "candidate_url",
            "reference_image_url",
            "match_status",
            "provider",
        ):
            if key in row and row[key] not in (None, ""):
                item[key] = _compact_trace_value(row[key], max_string=2400)
        if item:
            projected.append(item)
    return projected


def _project_tool_observation(
    tool_name: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expose completed observations while omitting media and raw HTML blobs."""

    if not result:
        return {}
    status = str(result.get("status", "")).strip()
    if tool_name == "text_search":
        queries = []
        for query_row in _rows(result.get("queries"))[:8]:
            query = {
                "query": _text(query_row.get("query"), limit=1200),
                "results": _project_search_rows(query_row.get("results")),
            }
            if query["query"] or query["results"]:
                queries.append(query)
        return {
            "status": status,
            "observation_status": _text(
                result.get("observation_status"),
                limit=80,
            ),
            "observation_note": _text(
                result.get("observation_note"),
                limit=800,
            ),
            "queries": queries,
            "error": _text(result.get("error"), limit=800),
        }
    if tool_name == "text_image_search":
        return {
            "status": status,
            "query": _text(result.get("query"), limit=1200),
            "observation_status": _text(
                result.get("observation_status"),
                limit=80,
            ),
            "observation_note": _text(
                result.get("observation_note"),
                limit=800,
            ),
            "results": _project_search_rows(result.get("results")),
            "candidate_page_urls": _string_list(
                result.get("candidate_page_urls"),
                limit=12,
            ),
            "reference_image_candidates": _string_list(
                result.get("reference_image_candidates"),
                limit=12,
            ),
            "error": _text(result.get("error"), limit=800),
        }
    if tool_name == "reverse_image_search":
        return {
            "status": status,
            "branch": _text(result.get("branch"), limit=80),
            "candidate_match_status": _text(
                result.get("candidate_match_status"),
                limit=120,
            ),
            "lens_results": _project_search_rows(result.get("lens_results")),
            "semantic_results": _project_search_rows(
                result.get("semantic_results")
            ),
            "candidate_page_urls": _string_list(
                result.get("candidate_page_urls"),
                limit=12,
            ),
            "reference_image_candidates": _string_list(
                result.get("reference_image_candidates"),
                limit=12,
            ),
            "error": _text(result.get("error"), limit=800),
        }
    if tool_name == "visit":
        records = []
        record_sources = list(_rows(result.get("evidence_records")))
        for visit in _rows(result.get("visits")):
            record_sources.extend(_rows(visit.get("evidence_records")))
        seen_records: set[str] = set()
        for row in record_sources:
            row_key = json.dumps(
                _compact_trace_value(row, max_string=8000),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if row_key in seen_records:
                continue
            seen_records.add(row_key)
            if len(records) >= 12:
                break
            records.append(
                {
                    key: _compact_trace_value(row[key], max_string=8000)
                    for key in (
                        "url",
                        "title",
                        "evidence",
                        "exact_text",
                        "evidence_context",
                        "context_spans",
                        "stance",
                        "directness",
                        "relevance",
                        "evidence_class",
                        "content_status",
                    )
                    if key in row and row[key] not in (None, "")
                }
            )
        return {
            "status": status,
            "url": _text(result.get("url"), limit=2400),
            "summary": _text(result.get("summary"), limit=3000),
            "content_status": _text(result.get("content_status"), limit=120),
            "evidence_records": records,
            "visited_pages": [
                {
                    "url": _text(visit.get("url"), limit=2400),
                    "summary": _text(visit.get("summary"), limit=1800),
                    "record_count": len(_rows(visit.get("evidence_records"))),
                }
                for visit in _rows(result.get("visits"))[:12]
            ],
            "error": _text(result.get("error"), limit=800),
        }
    return _compact_trace_value(dict(result))


def _raw_action_history(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Project raw unified-ReAct actions in actual episode order."""

    history: List[Dict[str, Any]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if str(step.get("action_type", "")).strip() != "tool_call":
            continue
        if str(step.get("stage", "")).strip() != "unified_react":
            continue
        tool_name = str(step.get("tool_name", "")).strip()
        if not tool_name or tool_name == "finish_investigation":
            continue
        tool_args = {
            str(key): _compact_trace_value(value, max_string=2400)
            for key, value in _mapping(step.get("tool_args")).items()
            if str(key).casefold() not in _TRACE_MEDIA_KEYS
        }
        metadata = _mapping(step.get("metadata"))
        result = _tool_result_mapping(step)
        observation_id = str(metadata.get("function_call_id", "")).strip()
        status = str(result.get("status", "")).strip().casefold() or "unknown"
        result_count = _tool_result_count(result)
        no_match = bool(
            status == "success"
            and tool_name
            in {"text_search", "text_image_search", "reverse_image_search"}
            and (
                str(result.get("observation_status", "")).strip().casefold()
                in {"empty_results", "no_match", "no_results"}
                or result_count == 0
            )
        )
        history.append(
            {
                "step_index": index,
                "turn": len(history) + 1,
                "observation_id": observation_id,
                "tool": tool_name,
                "thought": _text(step.get("thought"), limit=4000),
                "arguments": tool_args,
                "observation": _project_tool_observation(tool_name, result),
                "status": status,
                "result_count": result_count,
                "no_match": no_match,
                "tool_success": bool(
                    metadata.get(
                        "tool_success",
                        result.get("status") == "success",
                    )
                ),
            }
        )
        if len(history) >= 40:
            break
    return history


def _tool_result_count(result: Mapping[str, Any]) -> int:
    count = 0
    for key in (
        "results",
        "lens_results",
        "semantic_results",
        "reference_image_candidates",
        "candidate_page_urls",
        "evidence_records",
    ):
        count += len(_rows(result.get(key)))
    for query_row in _rows(result.get("queries")):
        count += len(_rows(query_row.get("results")))
    for visit in _rows(result.get("visits")):
        count += len(_rows(visit.get("evidence_records")))
    return count


def _string_list(value: Any, *, limit: int) -> List[str]:
    values = value if isinstance(value, list) else [value]
    return _unique(
        [
            str(item).strip()
            for item in values
            if isinstance(item, str) and item.strip()
        ],
        limit=limit,
    )


def _retrieval_history(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compress actual text retrieval into judge-visible process context."""

    history: List[Dict[str, Any]] = []
    for step in _rows(state.get("all_steps")):
        if str(step.get("action_type", "")).strip() != "tool_call":
            continue
        tool_name = str(step.get("tool_name", "")).strip()
        if tool_name not in {
            "text_search",
            "text_image_search",
            "reverse_image_search",
            "visit",
        }:
            continue
        tool_args = _mapping(step.get("tool_args"))
        result = _tool_result_mapping(step)
        status = str(result.get("status", "")).strip().lower()
        if tool_name == "text_search":
            query_rows = _rows(result.get("queries"))
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(
                        tool_args.get("goal"),
                        tool_args.get("question"),
                        tool_args.get("context"),
                        limit=800,
                    ),
                    "queries": _string_list(
                        tool_args.get("queries", tool_args.get("query", [])),
                        limit=3,
                    ),
                    "status": status or "unknown",
                    "result_count": sum(
                        len(_rows(row.get("results")))
                        for row in query_rows
                    ),
                    "search_error": _text(result.get("search_error"), result.get("error"), limit=500),
                }
            )
        elif tool_name == "visit":
            source_urls = _string_list(tool_args.get("url", []), limit=3)
            source_urls = _unique(
                [*source_urls, *_string_list(result.get("url", ""), limit=1)],
                limit=3,
            )
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(
                        tool_args.get("retrieval_goal"),
                        tool_args.get("goal"),
                        tool_args.get("question"),
                        tool_args.get("context"),
                        limit=800,
                    ),
                    "source_urls": source_urls,
                    "status": status or "unknown",
                    "page_text_extracted": bool(
                        _text(
                            result.get("evidence"),
                            result.get("exact_text"),
                            result.get("content"),
                        )
                    ),
                    "fetch_error": _text(result.get("error"), limit=500),
                }
            )
        elif tool_name == "text_image_search":
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(
                        tool_args.get("goal"),
                        tool_args.get("question"),
                        tool_args.get("context"),
                        limit=800,
                    ),
                    "queries": _string_list(
                        tool_args.get("query"),
                        limit=3,
                    ),
                    "status": status or "unknown",
                    "result_count": len(_rows(result.get("results"))),
                    "candidate_page_urls": _string_list(
                        result.get("candidate_page_urls"),
                        limit=3,
                    ),
                    "candidate_image_count": len(
                        _string_list(
                            result.get("reference_image_candidates"),
                            limit=6,
                        )
                    ),
                    "search_error": _text(
                        result.get("search_error"),
                        result.get("error"),
                        limit=500,
                    ),
                }
            )
        else:
            history.append(
                {
                    "tool": tool_name,
                    "goal": _text(
                        tool_args.get("goal"),
                        tool_args.get("question"),
                        tool_args.get("context"),
                        limit=800,
                    ),
                    "status": status or "unknown",
                    "result_count": sum(
                        len(_rows(result.get(key)))
                        for key in ("lens_results", "semantic_results")
                    ),
                    "candidate_page_urls": _string_list(
                        result.get("candidate_page_urls"),
                        limit=3,
                    ),
                    "candidate_image_count": len(
                        _string_list(
                            result.get("reference_image_candidates"),
                            limit=6,
                        )
                    ),
                    "search_error": _text(
                        result.get("search_error"),
                        result.get("error"),
                        limit=500,
                    ),
                }
            )
        if len(history) >= 24:
            break
    return history


def _rejection_history(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Project rejected policy turns for the post-rollout judge.

    This intentionally preserves the runtime's structural outcome and compact
    route context, rather than inferring a semantic error category from fragile
    message matching. The frozen judge decides whether the trajectory recovered
    or repeated the rejected behavior.
    """

    events: List[Dict[str, Any]] = []
    action_counts: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {}
    for index, step in enumerate(_rows(state.get("all_steps"))):
        action_type = str(step.get("action_type", "")).strip()
        metadata = _mapping(step.get("metadata"))
        rejected = (
            action_type in {"format_error", "output_rejected"}
            or str(metadata.get("error_class", "")).strip() == "protocol_error"
            or bool(str(metadata.get("rejection_reason", "")).strip())
        )
        if not rejected:
            continue
        tool_args = _mapping(step.get("tool_args"))
        tool_result = _tool_result_mapping(step)
        stage = str(step.get("stage", "")).strip() or "unknown"
        action_key = action_type or "rejected_step"
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        events.append(
            {
                "step_index": index,
                "stage": stage,
                "action_type": action_key,
                "tool": str(step.get("tool_name", "")).strip(),
                "error_class": str(metadata.get("error_class", "")).strip(),
                "reason": _text(
                    metadata.get("rejection_reason"),
                    tool_result.get("error"),
                    limit=1200,
                ),
                "goal": _text(
                    tool_args.get("retrieval_goal"),
                    tool_args.get("goal"),
                    limit=700,
                ),
                "queries": _string_list(
                    tool_args.get("queries", tool_args.get("query", [])),
                    limit=3,
                ),
            }
        )
        if len(events) >= 16:
            break
    return {
        "count": sum(action_counts.values()),
        "by_action_type": action_counts,
        "by_stage": stage_counts,
        "events": events,
    }


def build_sft_eligibility_input(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    image_path: Path | None = None,
) -> Dict[str, Any]:
    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    runtime_case = _mapping(state.get("runtime_case"))
    case_id = str(runtime_case.get("case_id") or trace.get("image_id") or "").strip()
    target = build_sft_target(gold)
    if case_id not in set(target.get("case_id_aliases", [])):
        raise ValueError(
            f"trace case_id {case_id!r} is not present in private target aliases"
        )

    basis = dict(_mapping(trace.get("verdict_basis")))
    if str(investigation.get("schema_version", "")).strip() != (
        REACT_RUNTIME_SCHEMA_VERSION
    ):
        raise ValueError("SFT eligibility requires the raw-history runtime schema")
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
    )

    raw_history = _raw_action_history(state)
    raw_observation_ids = _unique(
        item.get("observation_id")
        for item in raw_history
        if item.get("observation_id")
    )
    successful_observation_ids = _unique(
        item.get("observation_id")
        for item in raw_history
        if item.get("observation_id") and item.get("tool_success") is True
    )
    unsuccessful_observation_ids = _unique(
        item.get("observation_id")
        for item in raw_history
        if item.get("observation_id") and item.get("tool_success") is not True
    )
    judgment_observation_ids = _unique(
        judgment.get("verdict_observation_ids")
        or judgment.get("selected_observation_ids")
        or [],
        limit=12,
    )
    return {
        "schema_version": SFT_ELIGIBILITY_INPUT_VERSION,
        "case_id": case_id,
        "episode_id": str(trace.get("image_id") or state.get("image_id") or ""),
        "decision_policy_version": str(
            trace.get("decision_policy_version")
            or state.get("decision_policy_version")
            or ""
        ),
        "recorded_verdict": str(trace.get("verdict", "")),
        "termination": str(trace.get("termination", "")),
        "image": {
            "image_sha256": str(runtime_case.get("image_sha256", "")),
            "available_to_judge": bool(image_path and image_path.is_file()),
        },
        "candidate": {
            "runtime_mode": "raw_history",
            "recorded_verdict": str(
                judgment.get("verdict") or trace.get("verdict") or ""
            ),
            "final_fact_check_report": _mapping(
                judgment.get("fact_check_report")
                or trace.get("fact_check_report")
            ),
            "final_evidence_citations": [
                {
                    "evidence_id": str(item.get("evidence_id", "")),
                    "source_url": _text(item.get("source_url"), limit=2000),
                    "source_family": _text(item.get("source_family"), limit=300),
                    "evidence_kind": _text(item.get("evidence_kind"), limit=100),
                    "relation_stance": _text(
                        item.get("relation_stance"),
                        limit=100,
                    ),
                    "excerpt": _text(item.get("excerpt"), limit=2400),
                }
                for item in _rows(judgment.get("evidence_citations"))
                if str(item.get("evidence_id", "")).strip()
            ],
            "raw_observations": raw_history,
            "raw_observation_ids": raw_observation_ids,
            "successful_observation_ids": successful_observation_ids,
            "unsuccessful_observation_ids": unsuccessful_observation_ids,
            "verdict_observation_ids": judgment_observation_ids,
            "retrieval_history": _retrieval_history(state),
            "rejection_history": _rejection_history(state),
            "final_visual_audit": _mapping(state.get("final_visual_audit")),
            "verdict_target": _text(
                basis.get("verdict_target"),
                basis.get("objective"),
                limit=4000,
            ),
            "unresolved_gaps": _unique(
                basis.get("unresolved_gaps")
                or basis.get("open_questions")
                or [],
                limit=12,
            ),
        },
        "private_target": target,
    }


def classify_sft_audit_failures(
    failures: Iterable[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep non-safety audit findings from becoming an all-or-nothing SFT gate."""

    fatal: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    fatal_markers = (
        "ENGINEERING",
        "SOURCE_ACCESS",
        "POLICY_VIOLATION",
        "TRACE_CORRUPT",
        "MALFORMED",
        "TERMINATION_NOT_SUCCESS",
        "POST_VERDICT",
        "PROVIDER_FAILURE",
        "EVIDENCE_CALL_NOT_SUCCESSFUL",
        "INVALID_EVIDENCE",
    )
    for raw in failures:
        item = dict(raw)
        code = str(item.get("code", "")).upper()
        if any(marker in code for marker in fatal_markers):
            fatal.append(item)
        else:
            warnings.append(item)
    return fatal, warnings


def sft_eligibility_metrics(
    packet: Mapping[str, Any],
    judgment: SFTEligibilityJudgment | None,
) -> Dict[str, Any]:
    target = _mapping(packet.get("private_target"))
    candidate = _mapping(packet.get("candidate"))
    expected_verdict = str(target.get("expected_verdict", ""))
    recorded_verdict = str(packet.get("recorded_verdict", ""))
    image_available = bool(_mapping(packet.get("image")).get("available_to_judge"))
    expected_support = (
        "supports_real" if expected_verdict == "real" else "supports_fake"
    )
    raw_observations = _rows(candidate.get("raw_observations"))
    raw_by_id = {
        str(item.get("observation_id", "")).strip(): item
        for item in raw_observations
        if str(item.get("observation_id", "")).strip()
    }
    if str(candidate.get("runtime_mode", "")).strip() != "raw_history":
        raise ValueError("SFT metrics require a raw-history candidate")
    if judgment is None:
        judgment_values: Dict[str, Any] = {
            "target_scope": "unclear",
            "decision_support": "unclear",
            "retrieval_quality": "poor",
            "decisive_observation_ids": [],
            "supporting_observation_ids": [],
            "overclaiming": "major",
            "boundary_assessment": "major_issue",
            "trajectory_conduct": "unresolved",
            "confidence": 0.0,
            "explanation": "judge_not_run",
        }
    else:
        judgment_values = judgment.model_dump(mode="json")

    decisive_ids = _unique(
        judgment_values.get("decisive_observation_ids", []),
        limit=40,
    )
    supporting_ids = _unique(
        judgment_values.get("supporting_observation_ids", []),
        limit=40,
    )
    selected_ids = _unique([*decisive_ids, *supporting_ids], limit=40)
    selected_by_id = raw_by_id
    invalid_ids = [item for item in selected_ids if item not in selected_by_id]
    failed_ids = [
        item
        for item in selected_ids
        if item in selected_by_id
        and (
            not bool(
                selected_by_id[item].get(
                    "tool_success",
                    selected_by_id[item].get("successful_call", False),
                )
            )
        )
    ]
    valid_decisive_ids = [
        item
        for item in decisive_ids
        if item in selected_by_id
        and bool(
            selected_by_id[item].get(
                "tool_success",
                selected_by_id[item].get("successful_call", False),
            )
        )
    ]
    decisive_rows = [selected_by_id[item] for item in valid_decisive_ids]
    empty_search_only = bool(decisive_rows) and all(
        bool(item.get("no_match"))
        or (
            str(item.get("tool", item.get("tool_name", ""))).strip()
            in {"text_search", "text_image_search", "reverse_image_search"}
            and int(item.get("result_count", 0) or 0) == 0
        )
        for item in decisive_rows
    )
    target_scope = str(judgment_values.get("target_scope", "unclear"))
    decision_support = str(judgment_values.get("decision_support", "unclear"))
    retrieval_quality = str(judgment_values.get("retrieval_quality", "poor"))
    trajectory_conduct = str(
        judgment_values.get("trajectory_conduct", "unresolved")
    )
    fatal_errors = [
        *(
            ["image_unavailable_to_judge"]
            if not image_available
            else []
        ),
        *(
            ["invalid_judge_observation_ids"]
            if invalid_ids
            else []
        ),
        *(
            ["selected_observation_not_successful"]
            if failed_ids
            else []
        ),
        *(
            ["unrelated_image_fact"]
            if target_scope == "unrelated_fact"
            else []
        ),
        *(
            ["key_target_condition_unchecked"]
            if target_scope == "related_but_incomplete"
            else []
        ),
        *(
            ["major_overclaiming"]
            if judgment_values.get("overclaiming") == "major"
            else []
        ),
        *(
            ["poor_retrieval_quality"]
            if retrieval_quality == "poor"
            else []
        ),
        *(
            ["no_decisive_observation"]
            if not valid_decisive_ids
            else []
        ),
        *(
            ["empty_search_only_cannot_support_fake"]
            if expected_verdict == "fake" and empty_search_only
            else []
        ),
        *(
            [f"trajectory_conduct_{trajectory_conduct}"]
            if trajectory_conduct in {"degraded_repetition", "unresolved"}
            else []
        ),
    ]
    warnings: List[str] = []
    if target_scope == "unclear":
        warnings.append("target_scope_unclear")
    if decision_support in {"supporting_only", "insufficient", "unclear"}:
        warnings.append("decision_support_not_decisive")
    if retrieval_quality == "mixed":
        warnings.append("mixed_retrieval_quality")
    if judgment_values.get("overclaiming") == "minor":
        warnings.append("minor_overclaiming")
    if judgment_values.get("boundary_assessment") != "respected":
        warnings.append("boundary_warning")
    if trajectory_conduct == "recovered_minor":
        warnings.append("recovered_policy_rejection")
    basis_observation_ids = set(
        _unique(
            candidate.get("verdict_observation_ids")
            or candidate.get("successful_observation_ids")
            or [],
            limit=40,
        )
    )
    if basis_observation_ids and not basis_observation_ids.intersection(
        valid_decisive_ids
    ):
        warnings.append("decisive_observation_not_in_runtime_basis")
    raw_confidence = float(judgment_values.get("confidence", 0.0) or 0.0)
    normalized_confidence = max(0.0, min(1.0, raw_confidence))
    if normalized_confidence != raw_confidence:
        warnings.append("confidence_normalized_out_of_range")

    return {
        "expected_verdict": expected_verdict,
        "recorded_verdict": recorded_verdict,
        "expected_decision_support": expected_support,
        "verdict_correct": recorded_verdict == expected_verdict,
        "image_available": image_available,
        "target_scope": target_scope,
        "decision_support": decision_support,
        "retrieval_quality": retrieval_quality,
        "trajectory_conduct": trajectory_conduct,
        "overclaiming": str(judgment_values.get("overclaiming", "major")),
        "boundary_assessment": str(
            judgment_values.get("boundary_assessment", "major_issue")
        ),
        "runtime_mode": str(candidate.get("runtime_mode", "")),
        "decisive_observation_ids": valid_decisive_ids,
        "selected_observation_ids": selected_ids,
        "invalid_judge_observation_ids": invalid_ids,
        "failed_selected_observation_ids": failed_ids,
        "fatal_errors": _unique(fatal_errors, limit=20),
        "warnings": _unique(warnings, limit=20),
        "confidence": normalized_confidence,
        "raw_confidence": raw_confidence,
        "explanation": str(judgment_values.get("explanation", "")),
    }


def sft_eligibility_passes(
    metrics: Mapping[str, Any],
    *,
    strict_trace_audit_pass: bool,
    engineering_valid: bool,
    fatal_audit_errors: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Apply only safety gates; non-fatal strict-audit warnings do not veto SFT."""

    expected_support = str(metrics.get("expected_decision_support", ""))
    return bool(
        engineering_valid
        and metrics.get("verdict_correct") is True
        and metrics.get("target_scope")
        in {"direct_target", "decisive_subfact"}
        and metrics.get("decision_support") == expected_support
        and metrics.get("decisive_observation_ids")
        and not metrics.get("fatal_errors")
        and not list(fatal_audit_errors)
    )


class SFTEligibilityJudge:
    """Run one standalone structured ImageFact audit after a teacher rollout."""

    def __init__(
        self,
        backend: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        enable_thinking: bool | None = None,
    ) -> None:
        self._delegate = SemanticRewardJudge(
            backend,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        self.provider = self._delegate.provider
        self.model = self._delegate.model
        self.max_tokens = int(max_tokens)
        self.enable_thinking = enable_thinking

    @property
    def generation_identity(self) -> str:
        thinking = (
            "unset" if self.enable_thinking is None else str(bool(self.enable_thinking)).lower()
        )
        wire_api = str(getattr(self._delegate.backend, "wire_api", "default"))
        base_url = str(getattr(self._delegate.backend, "base_url", "default"))
        return (
            f"{SFT_ELIGIBILITY_GENERATION_VERSION}:provider={self.provider}:"
            f"model={self.model}:wire={wire_api}:base={base_url}:"
            f"max_tokens={self.max_tokens}:thinking={thinking}"
        )

    async def judge(
        self,
        packet: Mapping[str, Any],
        *,
        image_path: Path | None = None,
    ) -> tuple[SFTEligibilityJudgment, Dict[str, Any]]:
        image_data_url: str | None = None
        image_view: Dict[str, Any] | None = None
        if image_path is not None:
            image_data_url, image_view = controlled_image_to_data_url(
                str(image_path),
            )
        call = await self._delegate._call(
            system_prompt=SFT_ELIGIBILITY_SYSTEM_PROMPT,
            prompt_version=SFT_ELIGIBILITY_PROMPT_VERSION,
            payload=packet,
            response_model=SFTEligibilityJudgment,
            image_data_url=image_data_url,
        )
        return (
            SFTEligibilityJudgment.model_validate(call.parsed),
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_versions": [SFT_ELIGIBILITY_PROMPT_VERSION],
                "generation_version": SFT_ELIGIBILITY_GENERATION_VERSION,
                "max_tokens": self.max_tokens,
                "image_view": image_view,
                "calls": [call.audit],
            },
        )


def build_sft_eligibility_artifact(
    *,
    trace: Mapping[str, Any],
    trace_sha256: str,
    packet: Mapping[str, Any],
    judgment: SFTEligibilityJudgment | None,
    judge_audit: Mapping[str, Any] | None,
    strict_trace_audit_pass: bool,
    strict_trace_audit_failures: Iterable[Mapping[str, Any]] = (),
    strict_trace_audit_warnings: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    engineering_valid = bool(
        str(trace.get("termination", "")) == "success"
        and str(trace.get("verdict", "")) in {"real", "fake"}
    )
    audit_failures = [dict(item) for item in strict_trace_audit_failures]
    explicit_audit_warnings = [
        dict(item) for item in strict_trace_audit_warnings
    ]
    fatal_audit_errors, failure_warnings = classify_sft_audit_failures(
        audit_failures
    )
    audit_warnings = [*failure_warnings, *explicit_audit_warnings]
    metrics = sft_eligibility_metrics(packet, judgment)
    metrics["fatal_audit_errors"] = fatal_audit_errors
    metrics["audit_warnings"] = audit_warnings
    passed = bool(
        judgment is not None
        and sft_eligibility_passes(
            metrics,
            strict_trace_audit_pass=strict_trace_audit_pass,
            engineering_valid=engineering_valid,
            fatal_audit_errors=fatal_audit_errors,
        )
    )
    if not engineering_valid:
        metrics["fatal_errors"] = _unique(
            [*metrics.get("fatal_errors", []), "engineering_invalid"],
            limit=20,
        )
    core = {
        "schema_version": SFT_ELIGIBILITY_SCHEMA_VERSION,
        "postprocess_version": SFT_ELIGIBILITY_POSTPROCESS_VERSION,
        "case_id": str(packet.get("case_id", "")),
        "episode_id": str(packet.get("episode_id", "")),
        "source_trace": {
            "sha256": trace_sha256,
            "decision_policy_version": str(
                packet.get("decision_policy_version", "")
            ),
        },
        "eligibility_input": {
            "schema_version": packet.get("schema_version"),
            "sha256": sha256_json(packet),
            "image": packet.get("image"),
        },
        "judge": dict(judge_audit or {}),
        "structured_judgment": (
            judgment.model_dump(mode="json") if judgment is not None else None
        ),
        "metrics": metrics,
        "gates": {
            "strict_trace_audit_pass": bool(strict_trace_audit_pass),
            "strict_trace_audit_failures": audit_failures,
            "strict_trace_audit_warnings": explicit_audit_warnings,
            "fatal_audit_errors": fatal_audit_errors,
            "audit_warnings": audit_warnings,
            "engineering_valid": engineering_valid,
            "sft_eligibility_pass": passed,
        },
    }
    return {
        **core,
        "artifact_id": f"sha256:{sha256_json(core)}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def sft_eligibility_cache_key(
    *,
    trace_sha256: str,
    packet: Mapping[str, Any],
    provider: str,
    model: str,
    generation_version: str,
) -> str:
    return sha256_json(
        {
            "trace_sha256": trace_sha256,
            "eligibility_input_sha256": sha256_json(packet),
            "provider": provider,
            "model": model,
            "generation_version": generation_version,
            "postprocess_version": SFT_ELIGIBILITY_POSTPROCESS_VERSION,
            "prompt_versions": [SFT_ELIGIBILITY_PROMPT_VERSION],
        }
    )
