"""Small persistent asyncio runtime for synchronous tool adapters."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable, Optional


class PersistentAsyncRuntime:
    """Run concurrent coroutines on one long-lived asyncio event loop.

    Synchronous tools call :meth:`run` from worker threads.  A single event loop
    keeps one async HTTP transport reusable while ``run_coroutine_threadsafe``
    still allows independent requests to overlap.
    """

    def __init__(self, *, thread_name: str) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: Optional[BaseException] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_name = thread_name

    def run(self, coroutine: Awaitable[Any]) -> Any:
        self._ensure_started()
        with self._lock:
            if self._closed:
                raise RuntimeError("Persistent asyncio runtime is closed.")
            loop = self._loop
        if loop is None:
            raise RuntimeError("Persistent asyncio runtime is not ready.")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(
        self,
        cleanup_factory: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            loop = self._loop

        if thread is None:
            return
        self._ready.wait()
        cleanup_error: Optional[BaseException] = None
        if loop is not None and not loop.is_closed():
            try:
                if cleanup_factory is not None:
                    cleanup = cleanup_factory()
                    asyncio.run_coroutine_threadsafe(cleanup, loop).result(
                        timeout=10
                    )
            except BaseException as exc:
                # Even if provider cleanup fails or times out, the loop must
                # still be stopped.  Otherwise the daemon thread and any
                # transport it owns survive until process exit.
                cleanup_error = exc
            finally:
                loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        if thread.is_alive():
            close_error = RuntimeError(
                "Persistent asyncio runtime did not close cleanly."
            )
            if cleanup_error is not None:
                close_error.__cause__ = cleanup_error
            raise close_error
        if cleanup_error is not None:
            raise cleanup_error

    def _ensure_started(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Persistent asyncio runtime is closed.")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name=self._thread_name,
                    daemon=True,
                )
                self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("Persistent asyncio runtime failed to start.") from (
                self._startup_error
            )

    def _run_loop(self) -> None:
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            raise
        finally:
            if loop is not None:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()
