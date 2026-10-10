# -*- coding: utf-8 -*-
"""Unified ReAct image-only workflow: the single entry point for factual investigation.

Usage:
    from eviscene.workflow import VerificationWorkflow, WorkflowConfig

    workflow = VerificationWorkflow(WorkflowConfig())
    result = await workflow.run_single("path/to/image.jpg")
"""
from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

# Limit thread usage to prevent memory explosion
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from eviscene.orchestrator.runtime_case import image_sha256, verify_case_image
from eviscene.orchestrator.pipeline import Orchestrator
from eviscene.orchestrator.runtime_events import (
    CANONICAL_TRACE_SCHEMA_VERSION,
    CaseRuntimeStore,
    atomic_write_json,
    bind_case_runtime_store,
    reset_case_runtime_store,
)
from eviscene.orchestrator.state import ImageOnlyRuntimeCase
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.tool_ablation import ReactToolFamilyConfig
from eviscene.provider_profiles import resolve_provider_settings
from eviscene.redaction import sanitize_for_persistence
from eviscene.storage import default_trace_dir


AGENT_DECISION_POLICY_VERSION = "unified-react-v1"


@dataclass
class WorkflowConfig:
    """Configuration for the verification workflow."""

    # LLM settings
    profile_id: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    vlm_provider: Optional[str] = None  # Defaults to provider
    vlm_model: Optional[str] = None  # Defaults to model_name
    llm_wire_api: Optional[str] = None
    vlm_wire_api: Optional[str] = None
    llm_base_url: Optional[str] = field(default=None, init=False)
    vlm_base_url: Optional[str] = field(default=None, init=False)
    image_access_mode: Literal["direct_multimodal", "separate_vlm"] = (
        "direct_multimodal"
    )
    temperature: float = 0.0
    max_tokens: int = 8192
    sampling_seed: Optional[int] = None

    # External-evidence tool-family ablations. Direct visual operations and
    # finish_investigation remain available in every configuration.
    enable_web_search: bool = True
    enable_image_retrieval: bool = True
    enable_evidence_inspection: bool = True

    # Runtime settings
    timeout: float = 1800.0
    # Optional previous run directory whose completed tool results may be
    # reused when a case is retried.
    resume_from: Optional[str] = None

    # Output
    output_dir: str = field(default_factory=default_trace_dir)
    save_traces: bool = True
    source_access_policy: Optional[SourceAccessPolicy] = None
    decision_policy_version: str = AGENT_DECISION_POLICY_VERSION

    def __post_init__(self) -> None:
        resolved = resolve_provider_settings(
            profile_id=self.profile_id,
            provider=self.provider,
            model_name=self.model_name,
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            llm_wire_api=self.llm_wire_api,
            vlm_wire_api=self.vlm_wire_api,
        )
        self.profile_id = resolved.profile_id
        self.provider = resolved.provider
        self.model_name = resolved.model_name
        self.vlm_provider = resolved.vlm_provider
        self.vlm_model = resolved.vlm_model
        self.llm_wire_api = resolved.llm_wire_api
        self.vlm_wire_api = resolved.vlm_wire_api
        self.llm_base_url = resolved.base_url
        self.vlm_base_url = resolved.vlm_base_url
        configured_image_access_mode = str(self.image_access_mode).strip().lower()
        if configured_image_access_mode not in {
            "direct_multimodal",
            "separate_vlm",
        }:
            raise ValueError(
                "image_access_mode must be 'direct_multimodal' or 'separate_vlm'"
            )
        self.image_access_mode = configured_image_access_mode  # type: ignore[assignment]
        if self.decision_policy_version != AGENT_DECISION_POLICY_VERSION:
            raise ValueError(
                "Only the unified-react-v1 agent policy is supported by the "
                "current workflow."
            )

    @property
    def react_tool_family_config(self) -> ReactToolFamilyConfig:
        return ReactToolFamilyConfig(
            enable_web_search=self.enable_web_search,
            enable_image_retrieval=self.enable_image_retrieval,
            enable_evidence_inspection=self.enable_evidence_inspection,
        )


class VerificationWorkflow:
    """Main workflow for image factual verification."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self._orchestrator: Optional[Orchestrator] = None

    def _get_orchestrator(self, *, validate_startup: bool = True) -> Orchestrator:
        """Lazy init orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(
                provider=self.config.provider,
                model_name=self.config.model_name,
                vlm_provider=self.config.vlm_provider,
                vlm_model=self.config.vlm_model,
                llm_wire_api=self.config.llm_wire_api,
                vlm_wire_api=self.config.vlm_wire_api,
                llm_base_url=self.config.llm_base_url,
                vlm_base_url=self.config.vlm_base_url,
                image_access_mode=self.config.image_access_mode,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                sampling_seed=self.config.sampling_seed,
                tool_family_config=self.config.react_tool_family_config,
                source_access_policy=self.config.source_access_policy,
                validate_startup=validate_startup,
            )
        return self._orchestrator

    async def aclose(self) -> None:
        """Release resources owned by this workflow's orchestrator.

        ``run_batch`` deliberately creates an isolated child workflow per
        rollout so mutable investigation state and durable event streams cannot
        cross-contaminate.  Those children also own persistent HTTP transports
        and helper threads, so they must be closed as soon as the rollout
        reaches any terminal outcome rather than waiting for process exit.
        """

        orchestrator = self._orchestrator
        self._orchestrator = None
        if orchestrator is not None:
            await orchestrator.aclose()

    def _new_batch_child(self, config: WorkflowConfig) -> "VerificationWorkflow":
        """Construct one isolated rollout child.

        This narrow factory keeps the production isolation contract explicit and
        makes batch lifecycle tests independent of external providers.
        """

        return VerificationWorkflow(config)

    async def run_single(
        self,
        image_path: str,
        image_id: str = "",
        *,
        runtime_case: Optional[ImageOnlyRuntimeCase] = None,
    ) -> Dict[str, Any]:
        """Run one unified-ReAct image-only investigation.

        Non-image-only runtime inputs are intentionally unsupported. When no case is
        supplied, the workflow constructs the three-field public case locally.
        """
        if runtime_case is None:
            resolved_image_path = os.path.abspath(image_path)
            runtime_case = ImageOnlyRuntimeCase(
                case_id=image_id or os.path.basename(image_path) or image_path,
                image_path=resolved_image_path,
                image_sha256=image_sha256(resolved_image_path),
            )
            image_path = resolved_image_path
        elif not isinstance(runtime_case, ImageOnlyRuntimeCase):
            raise TypeError(
                "unified-react-v1 accepts ImageOnlyRuntimeCase only"
            )

        episode_id = image_id or runtime_case.case_id
        orchestrator = self._get_orchestrator(validate_startup=False)
        runtime_store = CaseRuntimeStore(
            self.config.output_dir,
            case_id=episode_id,
            resume_from=self.config.resume_from,
        )
        runtime_token = bind_case_runtime_store(runtime_store)
        try:
            verify_case_image(runtime_case, image_path)
            run_kwargs: Dict[str, Any] = {
                "decision_policy_version": self.config.decision_policy_version,
            }
            # The public single-rollout path keeps case_id and episode_id equal.
            # The public single-rollout path keeps case_id and episode_id equal.
            if episode_id != runtime_case.case_id:
                run_kwargs["episode_id"] = episode_id
            result = await orchestrator.run(
                image_path,
                runtime_case,
                **run_kwargs,
            )
        except Exception as exc:
            state = getattr(orchestrator, "last_state", None)
            error_result: Optional[Dict[str, Any]] = None
            if state is not None:
                error_result = {
                    "image_id": state.image_id,
                    "image_path": state.image_path,
                    "verdict": "error",
                    "confidence": 0.0,
                    "termination": state.termination or "error",
                    "error": " | ".join(state.errors) or str(exc),
                    "time_taken": state.stage_timings.get("total", 0.0),
                    "total_tool_calls": state.total_tool_calls,
                    "llm_api_calls": state.llm_api_calls,
                    "token_usage": state.token_usage,
                    "state": state.to_dict(),
                }
            if self.config.save_traces and state is not None:
                runtime_store.write_snapshot("engineering_error", state.to_dict())
                self._save_trace(
                    {
                        "image_id": state.image_id,
                        "image_path": state.image_path,
                        "input_mode": state.input_mode,
                        "decision_policy_version": state.decision_policy_version,
                        "verdict": "error",
                        "confidence": 0.0,
                        "overall_assessment": "Verification terminated with an engineering error.",
                        "state": state.to_dict(),
                        "termination": "error",
                        "time_taken": state.stage_timings.get("total", 0.0),
                        "token_usage": state.token_usage,
                        "total_tool_calls": state.total_tool_calls,
                        "llm_api_calls": state.llm_api_calls,
                        "error": " | ".join(state.errors),
                    }
                )
            if error_result is not None:
                setattr(exc, "_eviscene_result", error_result)
            raise
        finally:
            reset_case_runtime_store(runtime_token)

        # Save trace if configured
        if self.config.save_traces:
            runtime_store.write_snapshot("final_state", result.get("state", {}))
            self._save_trace(result)

        return result

    async def run_batch(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
        runtime_cases: Optional[List[Optional[ImageOnlyRuntimeCase]]] = None,
        sampling_seeds: Optional[List[Optional[int]]] = None,
        concurrency: int = 1,
        on_result: Optional[
            Callable[[int, Dict[str, Any]], Optional[Awaitable[None]]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Run verification on multiple images.

        Args:
            image_paths: List of image file paths.
            image_ids: Optional list of identifiers.
            concurrency: Max concurrent verifications.

        Returns:
            List of results.
        """
        if image_ids is None:
            image_ids = [os.path.basename(p) for p in image_paths]
        if runtime_cases is None:
            runtime_cases = [None] * len(image_paths)
        if len(runtime_cases) != len(image_paths):
            raise ValueError("runtime_cases must match image_paths length")
        if sampling_seeds is None:
            sampling_seeds = [self.config.sampling_seed] * len(image_paths)
        if len(sampling_seeds) != len(image_paths):
            raise ValueError("sampling_seeds must match image_paths length")

        semaphore = asyncio.Semaphore(concurrency)

        async def _verify(
            path: str,
            img_id: str,
            runtime_case: Optional[ImageOnlyRuntimeCase],
            sampling_seed: Optional[int],
        ) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # Honor an explicitly overridden instance method.
                    if "run_single" in self.__dict__:
                        return await self.run_single(
                            path,
                            img_id,
                            runtime_case=runtime_case,
                        )
                    # Each rollout owns its mutable state; only content-addressed caches are shared.
                    # Copy resolved settings directly: dataclasses.replace would re-run profile validation.
                    child_config = copy.copy(self.config)
                    child_config.sampling_seed = sampling_seed
                    child = self._new_batch_child(child_config)
                    try:
                        return await child.run_single(
                            path,
                            img_id,
                            runtime_case=runtime_case,
                        )
                    finally:
                        await child.aclose()
                except Exception as exc:
                    error_result = getattr(exc, "_eviscene_result", None)
                    if isinstance(error_result, dict):
                        return error_result
                    raise

        async def _indexed(
            index: int,
            path: str,
            img_id: str,
            runtime_case: Optional[ImageOnlyRuntimeCase],
            sampling_seed: Optional[int],
        ) -> tuple[int, Dict[str, Any]]:
            try:
                value = await _verify(path, img_id, runtime_case, sampling_seed)
            except Exception as error:
                value = {
                    "image_id": img_id,
                    "image_path": path,
                    "verdict": "error",
                    "termination": "error",
                    "error": str(error),
                }
            return index, value

        tasks = [
            asyncio.create_task(
                _indexed(index, path, img_id, runtime_case, sampling_seed)
            )
            for index, (path, img_id, runtime_case, sampling_seed) in enumerate(
                zip(image_paths, image_ids, runtime_cases, sampling_seeds)
            )
        ]
        final: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        try:
            for task in asyncio.as_completed(tasks):
                index, value = await task
                final[index] = value
                if on_result is not None:
                    callback = on_result(index, value)
                    if callback is not None:
                        await callback
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if any(value is None for value in final):
            raise RuntimeError("batch completed without one result per input")
        return [value for value in final if value is not None]

    def _save_trace(self, result: Dict[str, Any]) -> None:
        """Save the canonical verification trace to the output directory."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        image_id = result.get("image_id", "unknown")
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in image_id)
        trace_path = os.path.join(self.config.output_dir, f"{safe_id}.json")

        # Remove non-serializable fields
        serializable = sanitize_for_persistence({
            k: v for k, v in result.items()
            if k != "state" or isinstance(v, dict)
        })
        serializable.setdefault("schema_version", CANONICAL_TRACE_SCHEMA_VERSION)
        atomic_write_json(trace_path, serializable)
