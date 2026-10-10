from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from eviscene.orchestrator.source_access import SourceAccessPolicy


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @abstractmethod
    def call(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def set_source_access_policy(self, policy: SourceAccessPolicy) -> None:
        """Bind a runtime retrieval policy without exposing it in the tool schema."""

        self.source_access_policy = policy
        for attribute in ("client", "browse_client"):
            child = getattr(self, attribute, None)
            setter = getattr(child, "set_source_access_policy", None)
            if callable(setter):
                setter(policy)
