from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from eviscene.integrations.clock.system_clock import SystemClockClient
from eviscene.tools.base import BaseTool


@dataclass
class CurrentTimeTool(BaseTool):
    """Return runtime current time from the system clock."""

    client: SystemClockClient | None = None
    name: str = "current_time"
    description: str = "Return the current runtime date, datetime, and timezone from the system clock."
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = SystemClockClient()

    def get_current_time(self) -> Dict[str, str]:
        return {"status": "success", **self.client.now()}

    def call(self, params: Dict[str, str]) -> Dict[str, str]:
        return self.get_current_time()
