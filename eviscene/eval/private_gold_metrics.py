"""Post-hoc metrics for direct-QA and evidence-bearing Agent audits.

Both modes share the private-gold verdict check and the same three reported
category names. Direct QA has no retrieved Evidence, so its first category is
the judge's ``quality_bucket=strong`` answer-quality assessment. Agent has a
persisted basis, selected Evidence, and report; its first category therefore
requires the judge's ``reason_quality=decisive_and_grounded`` assessment.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


PRIVATE_GOLD_BUCKETS = (
    "correct_point_with_strong_evidence",
    "correct_verdict_insufficient_evidence",
    "wrong_verdict",
)
KNOWN_QUALITY_BUCKETS = frozenset(
    {"strong", "usable", "rejected", "not_auditable"}
)


def private_gold_category(
    row: Mapping[str, Any],
) -> str | None:
    """Return the shared three-way category for one auditable judge row.

    ``None`` means the row cannot participate in the three-way comparison,
    typically because the private gold is not explicitly auditable or the
    judge failed.
    A correct verdict with a non-strong quality bucket is intentionally the
    second category, including ``usable`` and any other judge downgrade.
    """

    if row.get("private_gold_auditable") is not True:
        return None
    if row.get("status") != "completed":
        return None
    if row.get("verdict_matches_gold") is True:
        if row.get("quality_bucket") not in KNOWN_QUALITY_BUCKETS:
            return None
        if row.get("quality_bucket") == "strong":
            return PRIVATE_GOLD_BUCKETS[0]
        return PRIVATE_GOLD_BUCKETS[1]
    if row.get("verdict_matches_gold") is False:
        return PRIVATE_GOLD_BUCKETS[2]
    return None


def agent_private_gold_category(
    row: Mapping[str, Any],
) -> str | None:
    """Return the Agent reporting category under the evidence-aware standard."""

    if row.get("private_gold_auditable") is not True:
        return None
    if row.get("status") != "completed":
        return None
    if row.get("verdict_matches_gold") is True:
        if row.get("reason_quality") == "decisive_and_grounded":
            return PRIVATE_GOLD_BUCKETS[0]
        return PRIVATE_GOLD_BUCKETS[1]
    if row.get("verdict_matches_gold") is False:
        return PRIVATE_GOLD_BUCKETS[2]
    return None


def private_gold_category_counts(
    rows: list[Mapping[str, Any]],
) -> dict[str, int]:
    """Count the shared categories, retaining zero-valued keys."""

    counts = Counter(
        category
        for row in rows
        if (category := private_gold_category(row)) is not None
    )
    return {bucket: int(counts.get(bucket, 0)) for bucket in PRIVATE_GOLD_BUCKETS}


def agent_private_gold_category_counts(
    rows: list[Mapping[str, Any]],
) -> dict[str, int]:
    """Count Agent categories with actual evidence/report grounding required."""

    counts = Counter(
        category
        for row in rows
        if (category := agent_private_gold_category(row)) is not None
    )
    return {bucket: int(counts.get(bucket, 0)) for bucket in PRIVATE_GOLD_BUCKETS}


def private_gold_audit_summary(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the common summary used for Agent and direct-QA audits."""

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    quality_counts = Counter(
        str(row.get("quality_bucket", "unknown")) for row in rows
    )
    categorized = sum(
        private_gold_category(row) is not None for row in rows
    )
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "completed_count": sum(
            row.get("status") == "completed" for row in rows
        ),
        "private_gold_auditable_count": sum(
            row.get("private_gold_auditable") is True for row in rows
        ),
        "verdict_matches_gold_count": sum(
            row.get("verdict_matches_gold") is True for row in rows
        ),
        "quality_buckets": dict(quality_counts),
        "private_gold_categories": private_gold_category_counts(rows),
        "uncategorized_count": len(rows) - categorized,
    }


def agent_private_gold_audit_summary(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the Agent summary using its evidence-aware primary categories."""

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    quality_counts = Counter(
        str(row.get("quality_bucket", "unknown")) for row in rows
    )
    categorized = sum(
        agent_private_gold_category(row) is not None for row in rows
    )
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "completed_count": sum(
            row.get("status") == "completed" for row in rows
        ),
        "private_gold_auditable_count": sum(
            row.get("private_gold_auditable") is True for row in rows
        ),
        "verdict_matches_gold_count": sum(
            row.get("verdict_matches_gold") is True for row in rows
        ),
        "quality_buckets": dict(quality_counts),
        "private_gold_categories": agent_private_gold_category_counts(rows),
        "uncategorized_count": len(rows) - categorized,
    }


def annotate_private_gold_category(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an audit row and attach the shared category for downstream reports."""

    annotated = dict(row)
    category = private_gold_category(row)
    if category is not None:
        annotated["private_gold_category"] = category
    return annotated


def annotate_agent_private_gold_category(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy one Agent audit row and attach the evidence-aware primary bucket."""

    annotated = dict(row)
    category = agent_private_gold_category(row)
    if category is not None:
        annotated["private_gold_category"] = category
    return annotated
