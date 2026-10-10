"""Shared qualification rules for Evidence used by semantic decisions."""

from __future__ import annotations

import re
from typing import Any, Mapping


QUALIFIED_EVIDENCE_QUALITIES = frozenset({"strong", "moderate"})
_NON_BLOCKING_RISK_FLAGS = frozenset({"user_generated_content"})
_EDIT_DIFFERENCE_TYPES = frozenset({"addition", "removal", "modification"})
_EDIT_SIGNIFICANCE_TO_STRENGTH = {
    "low": "weak",
    "medium": "moderate",
    "high": "strong",
}
_EDIT_STRENGTH_ORDER = ("none", "weak", "moderate", "strong")
_SAME_CAPTURE_WORLD_CONTEXT_HINTS = frozenset(
    {
        "attributed",
        "attribution",
        "author",
        "caption",
        "created",
        "creator",
        "date",
        "dated",
        "event",
        "location",
        "located",
        "occurred",
        "place",
        "provenance",
        "published",
        "record",
        "source",
        "time",
        "venue",
        "where",
        "when",
    }
)
_SAME_CAPTURE_VISUAL_CONTENT_HINTS = frozenset(
    {
        "appear",
        "appears",
        "contain",
        "contains",
        "depict",
        "depicts",
        "hull",
        "identify",
        "identified",
        "identity",
        "label",
        "logo",
        "marking",
        "markings",
        "object",
        "read",
        "reads",
        "relation",
        "scene",
        "ship",
        "show",
        "shows",
        "text",
        "vehicle",
        "vessel",
        "visible",
    }
)


def _origin_type(fact: Any) -> str:
    origin = evidence_value(fact, "origin", None)
    if isinstance(origin, Mapping):
        return str(origin.get("type", "")).strip()
    return str(getattr(origin, "type", "")).strip()


def _semantic_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value or "").casefold())
        if token
    }


def derive_edit_evidence_summary(differences: Any) -> tuple[bool, str]:
    """Derive the canonical edit summary from typed comparison differences.

    ``edit_evidence_present`` and ``edit_evidence_strength`` are runtime-owned
    fields.  They must be derived identically when a live tool result is being
    normalized and when an older trace is replayed into Evidence.  In
    particular, a missing legacy summary must not silently turn a typed edit
    difference into ``False``.
    """

    if not isinstance(differences, (list, tuple)):
        return False, "none"
    edit_items = [
        item
        for item in differences
        if isinstance(item, Mapping)
        and str(item.get("type", "")).strip().lower()
        in _EDIT_DIFFERENCE_TYPES
    ]
    strengths = [
        _EDIT_SIGNIFICANCE_TO_STRENGTH.get(
            str(item.get("significance", "")).strip().lower()
        )
        for item in edit_items
    ]
    strengths = [item for item in strengths if item is not None]
    return bool(edit_items), max(
        strengths,
        key=_EDIT_STRENGTH_ORDER.index,
        default="none",
    )


def same_capture_can_support_visual_claim(fact: Any) -> bool:
    """Return whether a same-capture reference can support this visual claim.

    The Agent may author free-form predicate labels such as ``depicts_vessel`` or
    ``shows_ship``.  Runtime must not depend on one exact label such as
    ``appears_to_depict``.  This gate instead asks whether the claim is
    image-grounded and visually inspectable.  Same-capture Evidence is still not
    enough for world-context claims such as location, event, date, authorship, or
    provenance; those require a source assertion or a paired chain.
    """

    if _origin_type(fact) not in {"input_image", "ocr"}:
        return False
    kind = str(evidence_value(fact, "kind", "")).strip()
    if kind not in {"attribute", "relation", "text_claim"}:
        return False
    predicate = str(evidence_value(fact, "predicate", "")).strip()
    tokens = _semantic_tokens(predicate)
    if tokens & _SAME_CAPTURE_WORLD_CONTEXT_HINTS:
        return False
    if kind in {"attribute", "text_claim"}:
        return True
    return bool(tokens & _SAME_CAPTURE_VISUAL_CONTENT_HINTS)


def evidence_value(evidence: Any, field: str, default: Any = None) -> Any:
    if isinstance(evidence, Mapping):
        return evidence.get(field, default)
    return getattr(evidence, field, default)


def evidence_direction_is_coherent(evidence: Any, stance: str) -> bool:
    """Return whether recorded Evidence metadata can carry ``stance``."""

    if str(evidence_value(evidence, "stance", "")) != stance:
        return False
    evidence_kind = str(evidence_value(evidence, "evidence_kind", ""))
    if evidence_kind == "web_span":
        expected_relation_stance = {
            "support": "supports",
            "refute": "contradicts",
        }.get(stance)
        return bool(
            expected_relation_stance
            and str(evidence_value(evidence, "claim_binding", ""))
            == "source_assertion"
            and str(evidence_value(evidence, "relation_scope", ""))
            == "same_relation"
            and str(evidence_value(evidence, "relation_stance", ""))
            == expected_relation_stance
        )
    if evidence_kind != "reference_comparison":
        return True

    same_capture = evidence_value(
        evidence,
        "same_capture_or_near_duplicate",
        False,
    ) is True
    different_capture = evidence_value(
        evidence,
        "likely_different_original_capture",
        False,
    ) is True
    edit_present = evidence_value(
        evidence,
        "edit_evidence_present",
        False,
    ) is True
    if stance == "refute":
        return same_capture and not different_capture and edit_present
    if stance == "support":
        return same_capture and not different_capture and not edit_present
    return False


def evidence_is_qualified(evidence: Any) -> bool:
    """Return whether Evidence may enter a material semantic decision."""

    hard_risk_flags = {
        str(item)
        for item in evidence_value(evidence, "risk_flags", []) or []
    } - _NON_BLOCKING_RISK_FLAGS
    return bool(
        str(evidence_value(evidence, "directness", "")) == "direct"
        and str(evidence_value(evidence, "quality", ""))
        in QUALIFIED_EVIDENCE_QUALITIES
        and not hard_risk_flags
    )


def evidence_is_qualified_for_stance(evidence: Any, stance: str) -> bool:
    """Return whether Evidence can support the requested directional chain.

    Qualification and direction are separate checks.  ``evidence_is_qualified``
    validates provenance/quality/risk, while ``evidence_direction_is_coherent``
    validates the recorded stance and any reference-comparison constraints.
    Keeping this small composition here lets tools, audits, and exporters use
    one semantic gate without duplicating either rule.
    """

    return bool(
        evidence_is_qualified(evidence)
        and evidence_direction_is_coherent(evidence, stance)
    )


def required_assessment_stances(assessment: str) -> frozenset[str]:
    """Return directional Evidence requirements for an internal assessment."""

    return {
        "supported": frozenset({"support"}),
        "refuted": frozenset({"refute"}),
        "conflicted": frozenset({"support", "refute"}),
        "insufficient": frozenset(),
        "unclear": frozenset(),
    }.get(str(assessment).strip().lower(), frozenset())


def reference_only_assessment_is_admissible(
    assessment: str,
    evidence_rows: list[Any],
) -> bool:
    """Preserve v3's conservative standalone reference-comparison boundary."""

    if assessment not in {"supported", "refuted"} or not evidence_rows:
        return True
    qualified_rows = [item for item in evidence_rows if evidence_is_qualified(item)]
    if not qualified_rows or any(
        str(evidence_value(item, "evidence_kind", ""))
        != "reference_comparison"
        for item in qualified_rows
    ):
        return True
    if assessment == "supported":
        return not any(
            evidence_value(item, "edit_evidence_present", False) is True
            for item in qualified_rows
        )
    return any(
        evidence_value(item, "same_capture_or_near_duplicate", False) is True
        and evidence_value(item, "likely_different_original_capture", False)
        is not True
        and evidence_value(item, "edit_evidence_present", False) is True
        for item in qualified_rows
    )
