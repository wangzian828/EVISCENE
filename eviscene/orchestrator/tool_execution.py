"""Small, auditable lifecycle records for tool actions.

The agent currently executes tools synchronously.  Keeping the lifecycle
separate from the tool-result contract lets us expose the pause/execute/resume
boundary now and add durable pending actions later without changing every
tool implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Literal, Optional, TypeVar
from uuid import uuid4


_ToolResult = TypeVar("_ToolResult")


ToolExecutionStatus = Literal[
    "running",
    "pending",
    "completed",
    "failed",
    "timed_out",
]


async def run_tool_with_timeout(
    awaitable: Awaitable[_ToolResult],
    *,
    timeout_seconds: float,
) -> _ToolResult:
    """Run one tool call without waiting on an uncancellable worker thread.

    ``asyncio.wait_for(asyncio.to_thread(...))`` can wait past its deadline while
    the executor thread is still blocked in a synchronous HTTP/provider call.
    The runtime needs to close the action lifecycle and hand the timeout back to
    the Agent immediately.  The underlying worker is cancelled when possible;
    tools must still enforce their own request deadlines for bounded cleanup.
    """

    task = asyncio.ensure_future(awaitable)
    done, pending = await asyncio.wait(
        {task},
        timeout=max(0.0, float(timeout_seconds)),
    )
    if pending:
        task.cancel()

        def consume_cancelled_task(completed: asyncio.Future[Any]) -> None:
            if completed.cancelled():
                return
            try:
                completed.exception()
            except BaseException:
                pass

        task.add_done_callback(consume_cancelled_task)
        raise asyncio.TimeoutError
    return next(iter(done)).result()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolActionRecord:
    """Trace-only state for one concrete tool execution."""

    action_id: str
    tool_name: str
    status: ToolExecutionStatus
    requested_at: str
    started_at: str
    completed_at: Optional[str] = None
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    continuation_id: Optional[str] = None
    error: str = ""

    @classmethod
    def start(
        cls,
        tool_name: str,
        *,
        timeout_seconds: Optional[float] = None,
        continuation_id: Optional[str] = None,
    ) -> "ToolActionRecord":
        now = _utc_now()
        return cls(
            action_id=f"tool-{uuid4().hex}",
            tool_name=tool_name,
            status="running",
            requested_at=now,
            started_at=now,
            timeout_seconds=timeout_seconds,
            continuation_id=continuation_id,
        )

    def finish(
        self,
        status: ToolExecutionStatus,
        *,
        error: str = "",
    ) -> Dict[str, Any]:
        if status in {"running", "pending"}:
            raise ValueError(
                f"a tool action cannot finish with status={status!r}"
            )
        self.status = status
        self.completed_at = _utc_now()
        self.error = error
        return self.to_dict()

    def mark_pending(self) -> Dict[str, Any]:
        """Persist a non-terminal action for a later external resume."""

        self.status = "pending"
        self.completed_at = None
        return self.to_dict()

    def resume(
        self,
        *,
        continuation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move a pending action back to running without changing its action id."""

        if self.status != "pending":
            raise ValueError(
                "only pending tool actions can be resumed"
            )
        self.status = "running"
        self.started_at = _utc_now()
        self.completed_at = None
        self.retry_count += 1
        if continuation_id is not None:
            self.continuation_id = continuation_id
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_action_id": self.action_id,
            "tool_name": self.tool_name,
            "tool_execution_status": self.status,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "continuation_id": self.continuation_id,
            **({"tool_error": self.error} if self.error else {}),
        }
