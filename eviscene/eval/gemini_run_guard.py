"""Cross-process guard for concurrent Gemini trajectory collection."""

from __future__ import annotations

from eviscene.environment import get_env

import os
from pathlib import Path
from typing import Optional

from eviscene.storage import runtime_lock_root


def _positive_env(name: str, default: int) -> int:
    raw = get_env(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _pid_is_alive(pid: str) -> bool:
    if not pid.isdigit() or int(pid) < 1:
        return False
    if int(pid) == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


class GeminiRunGuard:
    """Own the single machine-wide Gemini agent evaluation slot.

    The shell launcher owns this slot when it sets ``EVISCENE_GEMINI_LOCK_DIR``.
    Direct Python launches, including the archive ``run_cases`` runner, acquire
    the same lock themselves to prevent overlapping teacher batches.
    """

    def __init__(self, lock_dir: Optional[Path], *, acquired: bool) -> None:
        self.lock_dir = lock_dir
        self.acquired = acquired

    @classmethod
    def acquire(
        cls,
        *,
        provider: str,
        concurrency: int,
        run_id: str = "",
    ) -> "GeminiRunGuard":
        if str(provider).strip().lower() != "gemini":
            return cls(None, acquired=False)

        requested = int(concurrency)
        if requested < 1:
            raise ValueError("Gemini rollout concurrency must be at least 1")
        # Use the requested concurrency unless a machine-wide cap is configured.
        cap = _positive_env(
            "GEMINI_EVAL_MAX_CONCURRENCY",
            requested,
        )
        if requested > cap:
            raise ValueError(
                f"Gemini rollout concurrency {requested} exceeds the configured "
                f"cap {cap}"
            )

        configured_request_limit = _positive_env(
            "GEMINI_MAX_INFLIGHT_REQUESTS",
            requested,
        )
        effective_request_limit = min(configured_request_limit, cap)
        os.environ["GEMINI_MAX_INFLIGHT_REQUESTS"] = str(
            effective_request_limit
        )
        os.environ.setdefault("GEMINI_EVAL_MAX_CONCURRENCY", str(cap))

        inherited_lock = get_env("EVISCENE_GEMINI_LOCK_DIR", "").strip()
        if inherited_lock:
            # The caller owns the inherited lock; do not double-own it here.
            return cls(Path(inherited_lock), acquired=False)

        lock_dir = runtime_lock_root() / "gemini-agent-eval.lock"
        lock_dir.parent.mkdir(parents=True, exist_ok=True)

        for _attempt in range(2):
            try:
                lock_dir.mkdir()
            except FileExistsError:
                owner_file = lock_dir / "owner.env"
                owner_pid = ""
                if owner_file.is_file():
                    for line in owner_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines():
                        if line.startswith("owner_pid="):
                            owner_pid = line.split("=", 1)[1].strip()
                            break
                if _pid_is_alive(owner_pid):
                    raise RuntimeError(
                        "a Gemini evaluation is already active; refusing a "
                        "second launch to avoid provider overload"
                    )
                owner_file.unlink(missing_ok=True)
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue

            owner_file = lock_dir / "owner.env"
            owner_file.write_text(
                "\n".join(
                    [
                        f"owner_pid={os.getpid()}",
                        f"run_id={run_id}",
                        f"requested_concurrency={requested}",
                        f"request_limit={effective_request_limit}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return cls(lock_dir, acquired=True)

        raise RuntimeError(
            f"could not acquire the Gemini evaluation lock: {lock_dir}"
        )

    def release(self) -> None:
        if not self.acquired or self.lock_dir is None:
            return
        owner_file = self.lock_dir / "owner.env"
        owner_pid = ""
        if owner_file.is_file():
            for line in owner_file.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines():
                if line.startswith("owner_pid="):
                    owner_pid = line.split("=", 1)[1].strip()
                    break
        if owner_pid == str(os.getpid()):
            owner_file.unlink(missing_ok=True)
            try:
                self.lock_dir.rmdir()
            except OSError:
                pass
        self.acquired = False

    def __enter__(self) -> "GeminiRunGuard":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
