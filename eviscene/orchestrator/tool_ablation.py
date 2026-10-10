"""Tool-family switches for controlled external-evidence ablations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from eviscene.orchestrator.react_runtime import REACT_RUNTIME_TOOLS


TOOL_ABLATION_SCHEMA_VERSION = "ifv-react-tool-ablation-v1"

# Only tools exposed by the formal ReAct runtime belong here. Historical
# registry entries such as crop_and_search, recall_evidence, and read_evidence
# are intentionally absent.
EXTERNAL_EVIDENCE_TOOL_FAMILIES = {
    "web_search": frozenset({"text_search"}),
    "image_retrieval": frozenset(
        {"text_image_search", "reverse_image_search"}
    ),
    "evidence_inspection": frozenset(
        {"visit", "compare_with_reference"}
    ),
}


@dataclass(frozen=True)
class ReactToolFamilyConfig:
    """Positive capability switches for the formal ReAct tool contract."""

    enable_web_search: bool = True
    enable_image_retrieval: bool = True
    enable_evidence_inspection: bool = True

    @property
    def disabled_tool_names(self) -> frozenset[str]:
        disabled: set[str] = set()
        if not self.enable_web_search:
            disabled.update(EXTERNAL_EVIDENCE_TOOL_FAMILIES["web_search"])
        if not self.enable_image_retrieval:
            disabled.update(EXTERNAL_EVIDENCE_TOOL_FAMILIES["image_retrieval"])
        if not self.enable_evidence_inspection:
            disabled.update(
                EXTERNAL_EVIDENCE_TOOL_FAMILIES["evidence_inspection"]
            )
        return frozenset(disabled)

    @property
    def enabled_tool_names(self) -> tuple[str, ...]:
        disabled = self.disabled_tool_names
        return tuple(name for name in REACT_RUNTIME_TOOLS if name not in disabled)

    def to_manifest(self) -> Dict[str, Any]:
        disabled = self.disabled_tool_names
        enabled = self.enabled_tool_names
        return {
            "schema_version": TOOL_ABLATION_SCHEMA_VERSION,
            "web_search_enabled": self.enable_web_search,
            "image_retrieval_enabled": self.enable_image_retrieval,
            "evidence_inspection_enabled": self.enable_evidence_inspection,
            "disabled_runtime_tools": sorted(disabled),
            "exposed_runtime_tools": [*enabled, "finish_investigation"],
        }
