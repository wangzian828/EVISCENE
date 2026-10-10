from eviscene.orchestrator.investigation_models import DiscrepancyJudgment
from eviscene.orchestrator.state import (
    Entity,
    ImageOnlyRuntimeCase,
    PerceptionReport,
    TextRegion,
    VerificationState,
)
__all__ = [
    "Entity",
    "DiscrepancyJudgment",
    "ImageOnlyRuntimeCase",
    "PerceptionReport",
    "TextRegion",
    "VerificationState",
    "UnifiedReactState",
]


def __getattr__(name: str):
    """Load the active runtime state lazily to avoid tool import cycles."""

    if name == "UnifiedReactState":
        from eviscene.orchestrator.react_runtime import UnifiedReactState

        return UnifiedReactState
    raise AttributeError(name)
