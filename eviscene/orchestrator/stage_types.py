"""Records shared by stage execution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageStep:
    """A single stage step."""

    round: int = 0
    stage_name: str = ""
    thought: str = ""
    action_type: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    output: Optional[Dict[str, Any]] = None
    tokens: Dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "thought": 0}
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionSession:
    """State shared by policy stages in one Gemini Interactions investigation."""

    previous_interaction_id: Optional[str] = None
    pending_input: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
