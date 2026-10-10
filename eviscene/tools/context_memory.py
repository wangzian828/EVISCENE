"""Bounded access to the current case's immutable investigation archive."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from eviscene.orchestrator.runtime_events import current_case_runtime_store
from eviscene.tools.base import BaseTool


@dataclass
class RecallEvidenceTool(BaseTool):
    name: str = "recall_evidence"
    description: str = (
        "Find relevant candidates in this case's archived tool results. "
        "Candidates are memory only, not new Evidence; call read_evidence for exact content."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional exact filters such as tool, stage, task_id, "
                        "claim_ids, hypothesis_id, evidence_ids, or failure_ids."
                    ),
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "required": ["query"],
        }
    )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        store = current_case_runtime_store()
        if store is None:
            return {"status": "error", "error": "no active case archive"}
        return store.recall_archive(
            query=str(params.get("query", "")),
            filters=(
                dict(params.get("filters", {}) or {})
                if isinstance(params.get("filters", {}), dict)
                else {}
            ),
            top_k=int(params.get("top_k", 5) or 5),
        )


@dataclass
class ReadEvidenceTool(BaseTool):
    name: str = "read_evidence"
    description: str = (
        "Read exact paginated content and provenance for one archived memory_id. "
        "Use this after recall before relying on archived material."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "length": {"type": "integer", "minimum": 1, "maximum": 24000},
                "include_raw": {"type": "boolean"},
            },
            "required": ["memory_id"],
        }
    )

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        store = current_case_runtime_store()
        if store is None:
            return {"status": "error", "error": "no active case archive"}
        return store.read_archive_item(
            str(params.get("memory_id", "")),
            offset=int(params.get("offset", 0) or 0),
            length=int(params.get("length", 6000) or 6000),
            include_raw=bool(params.get("include_raw", True)),
        )
