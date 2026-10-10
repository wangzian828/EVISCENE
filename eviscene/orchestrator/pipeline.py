# -*- coding: utf-8 -*-
"""Raw-history image factual investigation orchestrator."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from eviscene.integrations.gemini import take_runtime_metrics
from eviscene.integrations.browse.jina_reader import resolve_browse_extract_provider
from eviscene.integrations.clock.system_clock import SystemClockClient
from eviscene.orchestrator.runtime_case import verify_case_image
from eviscene.orchestrator.runtime_events import CaseRuntimeStore, current_case_runtime_store
from eviscene.orchestrator.unified_prompts import (
    UNIFIED_JUDGMENT_PROMPT_VERSION,
    UNIFIED_JUDGMENT_SYSTEM_PROMPT,
    UNIFIED_REACT_PROMPT_VERSION,
    UNIFIED_REACT_SYSTEM_PROMPT,
)
from eviscene.orchestrator.investigation_models import (
    DiscrepancyJudgment,
    InvestigationSegmentOutput,
    RawHistoryJudgmentOutput,
)
from eviscene.orchestrator.llm_backend import APIBackend
from eviscene.integrations.llm.prefix_cache import new_case_cache_salt, new_rollout_cache_salts
from eviscene.orchestrator.stage_runner import (
    InteractionSession,
    StageRunner,
    StageStep,
)
from eviscene.orchestrator.tool_execution import run_tool_with_timeout
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.state import (
    ImageOnlyRuntimeCase,
    VerificationState,
)
from eviscene.orchestrator.tool_cache import (
    ToolResultCache,
    WEB_EVIDENCE_CONTRACT_VERSION,
)
from eviscene.orchestrator.tool_health import require_tools, summarize_health
from eviscene.orchestrator.tool_ablation import ReactToolFamilyConfig
from eviscene.orchestrator.tool_registry import (
    REQUIRED_TOOLS,
    build_all_tools_with_health,
)
from eviscene.orchestrator.tool_result import parse_tool_result, serialize_tool_result
from eviscene.orchestrator.react_runtime import (
    UnifiedReactState as RuntimeReactState,
    MAX_REACT_ACTIONS,
    REACT_RUNTIME_TOOL_CALL_LIMITS,
    UNIFIED_REACT_RUNTIME_POLICY_VERSION as UNIFIED_REACT_POLICY_VERSION,
    build_react_runtime_tools,
    compile_react_judgment_basis,
    new_unified_react_runtime_state,
    record_react_action,
    render_react_judgment_context,
    render_react_runtime_context,
    validate_react_action,
)
from eviscene.storage import default_tool_cache_dir
from eviscene.provider_profiles import resolve_provider_settings


MAX_MODEL_TASK_CHOICES_PER_ACTION = 3


class Orchestrator:
    """One-loop raw-history ReAct orchestrator."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        llm_wire_api: Optional[str] = None,
        vlm_wire_api: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        image_access_mode: str = "direct_multimodal",
        timeout: float = 1800.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        sampling_seed: Optional[int] = None,
        validate_startup: bool = True,
        source_access_policy: Optional[SourceAccessPolicy] = None,
        tool_family_config: Optional[ReactToolFamilyConfig] = None,
    ):
        settings = resolve_provider_settings(
            provider=provider, model_name=model_name,
            vlm_provider=vlm_provider, vlm_model=vlm_model,
            llm_wire_api=llm_wire_api, vlm_wire_api=vlm_wire_api,
        )
        provider, model_name = settings.provider, settings.model_name
        vlm_provider, vlm_model = settings.vlm_provider, settings.vlm_model
        llm_base_url = llm_base_url or settings.base_url
        vlm_base_url = vlm_base_url or settings.vlm_base_url
        llm_wire_api = llm_wire_api or settings.llm_wire_api
        vlm_wire_api = vlm_wire_api or settings.vlm_wire_api
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.vlm_provider = (vlm_provider or self.provider).lower().strip()
        self.vlm_model = vlm_model or model_name
        self.llm_base_url = llm_base_url
        self.vlm_base_url = vlm_base_url
        self.image_access_mode = str(
            image_access_mode or "direct_multimodal"
        ).strip().lower()
        if self.image_access_mode not in {
            "direct_multimodal",
            "separate_vlm",
        }:
            raise ValueError(
                "image_access_mode must be 'direct_multimodal' or 'separate_vlm'"
            )
        self.llm_wire_api = llm_wire_api or os.getenv("AGENT_LLM_WIRE_API")
        if self.llm_wire_api is None and self.provider == "gemini":
            self.llm_wire_api = os.getenv("GEMINI_WIRE_API")
        self.vlm_wire_api = vlm_wire_api or os.getenv("VISION_LLM_WIRE_API")
        if self.vlm_wire_api is None and self.vlm_provider == "gemini":
            self.vlm_wire_api = (
                os.getenv("GEMINI_VISION_WIRE_API")
                or os.getenv("GEMINI_WIRE_API")
            )
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.tool_family_config = tool_family_config or ReactToolFamilyConfig()
        self.enabled_runtime_tools = frozenset(
            self.tool_family_config.enabled_tool_names
        )
        self.disabled_runtime_tools = (
            self.tool_family_config.disabled_tool_names
        )
        self.timeout = timeout
        self.sampling_seed = sampling_seed
        self._sampling_request_counts: Dict[str, int] = {}
        self.tool_action_timeout_seconds = self._runtime_timeout(
            "AGENT_TOOL_ACTION_TIMEOUT_SECONDS",
            210.0,
        )
        self.stage_request_timeout_seconds = self._runtime_timeout(
            "AGENT_STAGE_REQUEST_TIMEOUT_SECONDS",
            (
                900.0
                if self.provider in {"gemini", "qwen_local", "lmdeploy"}
                else 300.0
            ),
        )
        cache_namespace = os.getenv("TOOL_CACHE_NAMESPACE", "").strip() or "|".join(
            [
                "tool-contract-v1",
                self.provider,
                model_name,
                self.vlm_provider,
                self.vlm_model,
                os.getenv("VISUAL_SEARCH_PROVIDER", "serper_lens"),
                os.getenv("IMAGE_UPLOAD_PROVIDER", "oss"),
                os.getenv("BROWSE_FETCH_PROVIDER", "jina"),
                os.getenv("PERCEPTION_CACHE_VERSION", "perception-v1"),
                os.getenv(
                    "OCR_CACHE_VERSION",
                    f"{os.getenv('OCR_BACKEND', 'baidu').strip().lower()}-v1",
                ),
                self.source_access_policy.cache_partition,
            ]
        )
        perception_cache_enabled = os.getenv(
            "PERCEPTION_CACHE_ENABLED",
            "0",
        ).strip().lower() in {"1", "true", "yes"}
        web_cache_enabled = os.getenv(
            "TOOL_CACHE_ENABLED",
            "0",
        ).strip().lower() in {"1", "true", "yes"}
        self.tool_cache = ToolResultCache(
            cache_dir=os.getenv("TOOL_CACHE_DIR", default_tool_cache_dir()),
            enabled=perception_cache_enabled or web_cache_enabled,
            ttl_seconds=float(os.getenv("TOOL_CACHE_TTL_SECONDS", "3600")),
            namespace=cache_namespace,
        )
        self.cacheable_tools = set()
        if perception_cache_enabled:
            self.cacheable_tools.update(
                {
                    "perceive_scene",
                    "ocr_with_position",
                }
            )
        if web_cache_enabled:
            self.cacheable_tools.update(
                {
                    "text_search",
                    "text_image_search",
                    "visit",
                    "crop_and_inspect",
                    "focused_visual_inspection",
                    "check_consistency",
                    "analyze_visual_anomalies",
                }
            )
        self.verification_tool_limits = dict(REACT_RUNTIME_TOOL_CALL_LIMITS)

        # ``run_batch`` creates one Orchestrator per rollout.  In the optional
        # Canary mode keeps each domain stable within one rollout while never
        # crossing a case boundary. Agent and page-extractor prompt families
        # deliberately use different private domains.
        rollout_cache_salt, browse_cache_salt = new_rollout_cache_salts()
        vision_cache_salt = new_case_cache_salt()
        while vision_cache_salt is not None and vision_cache_salt in {
            rollout_cache_salt, browse_cache_salt
        }:
            vision_cache_salt = new_case_cache_salt()
        self.prefix_cache_salt = (
            rollout_cache_salt
            if self.provider in {"qwen_local", "lmdeploy"}
            else None
        )
        self.browse_extract_provider = resolve_browse_extract_provider(
            self.provider
            if self.provider in {"qwen_local", "lmdeploy"}
            else "qwen_local"
        )
        reuse_local_policy_for_browse = (
            self.browse_extract_provider == self.provider
            and self.provider in {"qwen_local", "lmdeploy"}
        )
        self.browse_extract_prefix_cache_salt = (
            browse_cache_salt
            if self.browse_extract_provider in {"qwen_local", "lmdeploy"}
            else None
        )

        self.llm = APIBackend(
            provider=self.provider,
            model_name=model_name,
            base_url=self.llm_base_url,
            wire_api=self.llm_wire_api,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(
                os.getenv(
                    "AGENT_LLM_REQUEST_TIMEOUT_SECONDS",
                    (
                        "90"
                        if self.provider == "gemini"
                        else "900"
                        if self.provider in {"qwen_local", "lmdeploy"}
                        else "300"
                    ),
                )
            ),
            max_retries=int(
                os.getenv(
                    "AGENT_LLM_REQUEST_MAX_RETRIES",
                    "12" if self.provider == "gemini" else "0",
                )
            ),
            cache_salt=self.prefix_cache_salt,
        )
        self.all_tools, self.tool_health = build_all_tools_with_health(
            vlm_provider=self.vlm_provider,
            vlm_model=self.vlm_model,
            vlm_wire_api=self.vlm_wire_api,
            vlm_base_url=self.vlm_base_url,
            browse_extract_provider=self.browse_extract_provider,
            browse_extract_model=(
                self.model_name if reuse_local_policy_for_browse else None
            ),
            browse_extract_base_url=(
                self.llm_base_url if reuse_local_policy_for_browse else None
            ),
            browse_extract_wire_api=(
                self.llm_wire_api if reuse_local_policy_for_browse else None
            ),
            prefix_cache_salt=self.browse_extract_prefix_cache_salt,
            vision_prefix_cache_salt=(
                vision_cache_salt
                if self.vlm_provider in {"qwen_local", "lmdeploy"}
                else None
            ),
        )
        self.all_tools = {
            name: tool
            for name, tool in self.all_tools.items()
            if name not in self.disabled_runtime_tools
        }
        self.cacheable_tools.intersection_update(self.all_tools)
        self.verification_tool_limits = {
            name: limit
            for name, limit in self.verification_tool_limits.items()
            if name in self.enabled_runtime_tools
        }
        for tool in self.all_tools.values():
            setter = getattr(tool, "set_source_access_policy", None)
            if callable(setter):
                setter(self.source_access_policy)
        self.tool_health_summary = summarize_health(self.tool_health)
        if validate_startup:
            require_tools(
                self.tool_health,
                (
                    name
                    for name in REQUIRED_TOOLS
                    if name in self.enabled_runtime_tools
                ),
            )
            self._validate_startup_configuration()
        clock = SystemClockClient().now()
        self.date_prefix = (
            f"Current date: {clock['current_date']} ({clock['timezone']}). "
            "Use this runtime date for time-sensitive judgments instead of model memory.\n\n"
        )

    async def aclose(self) -> None:
        """Close per-rollout transports and helper runtimes.

        A batch uses one ``Orchestrator`` per isolated rollout.  Several tool
        wrappers intentionally share a persistent VLM client, Jina reader, and
        API backend within that rollout.  They are referenced from more than
        one wrapper, so collect resources by identity and close each exactly
        once when the rollout completes.
        """

        child_attributes = (
            "client",
            "vlm_client",
            "vlm_backend",
            "browse_client",
            "lens_client",
            "image_search_client",
            "visual_search_client",
            "candidate_reranker",
            "upload_client",
            "serper_lens_client",
            "zhipu_client",
            "baidu_client",
        )
        pending: list[Any] = [getattr(self, "llm", None)]
        pending.extend(getattr(self, "all_tools", {}).values())
        resources: list[Any] = []
        seen: set[int] = set()
        while pending:
            resource = pending.pop()
            if resource is None:
                continue
            resource_id = id(resource)
            if resource_id in seen:
                continue
            seen.add(resource_id)
            resources.append(resource)
            pending.extend(
                getattr(resource, name, None) for name in child_attributes
            )

        for resource in resources:
            async_close = getattr(resource, "aclose", None)
            sync_close = getattr(resource, "close", None)
            try:
                if callable(async_close):
                    result = async_close()
                    if inspect.isawaitable(result):
                        await result
                elif callable(sync_close):
                    await asyncio.to_thread(sync_close)
            except Exception:
                # Cleanup must not replace a completed rollout's actual result.
                # The resource's own atexit guard remains a final best-effort
                # fallback if a provider transport refuses to close here.
                continue

    def _validate_startup_configuration(self) -> None:
        tool_thinking_levels = {
            "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL": os.getenv(
                "GEMINI_VERIFICATION_FINAL_THINKING_LEVEL", "low"
            ),
            "GEMINI_BROWSE_THINKING_LEVEL": os.getenv(
                "GEMINI_BROWSE_THINKING_LEVEL", "low"
            ),
            "GEMINI_VISION_THINKING_LEVEL": os.getenv(
                "GEMINI_VISION_THINKING_LEVEL", "low"
            ),
        }
        for env_name, value in tool_thinking_levels.items():
            normalized_value = value.strip().lower()
            if normalized_value != "low":
                raise ValueError(
                    f"{env_name} must be 'low' for the active agent."
                )
        if self.provider == "gemini" and not self.llm.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini agent."
            )
        if self.vlm_provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ):
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini perception."
            )
        serper_tools = {
            "text_search",
            "text_image_search",
            "reverse_image_search",
        }
        if self.enabled_runtime_tools.intersection(serper_tools) and not (
            os.getenv("SERPER_API_KEY", "").strip()
            or os.getenv("SERPER_KEY_ID", "").strip()
        ):
            raise RuntimeError(
                "SERPER_API_KEY is required for text, image, and Lens search."
            )

        if (
            "visit" in self.enabled_runtime_tools
            and self.browse_extract_provider == "gemini"
            and not (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )
        ):
            raise RuntimeError(
                "Gemini credentials are required for BROWSE_EXTRACT_PROVIDER=gemini."
            )

        reverse_tool = self.all_tools.get("reverse_image_search")
        visual_client = getattr(reverse_tool, "visual_search_client", None)
        upload_client = getattr(visual_client, "upload_client", None)
        if upload_client is not None:
            upload_client.validate_configuration()
        visual_provider = str(getattr(visual_client, "provider", "serper_lens"))
        if visual_provider == "zhipu_image_search" and not os.getenv(
            "ZHIPU_API_KEY", ""
        ).strip():
            raise RuntimeError(
                "ZHIPU_API_KEY is required for VISUAL_SEARCH_PROVIDER=zhipu_image_search."
            )

    def _sp(self, stage_system_prompt: str) -> str:
        return self.date_prefix + stage_system_prompt

    def _uses_separate_vlm(self) -> bool:
        return getattr(self, "image_access_mode", "direct_multimodal") == (
            "separate_vlm"
        )

    def _main_llm_attaches_image(self) -> bool:
        return not self._uses_separate_vlm()

    async def run(
        self,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
        *,
        episode_id: Optional[str] = None,
        decision_policy_version: str = UNIFIED_REACT_POLICY_VERSION,
        runtime_store: Optional[CaseRuntimeStore] = None,
    ) -> Dict[str, Any]:
        """Run one explicit image-only decision policy without hidden fallback."""

        verify_case_image(runtime_case, image_path)
        runtime_store = runtime_store or current_case_runtime_store()
        state = VerificationState(
            image_path=image_path,
            image_id=episode_id or runtime_case.case_id,
            runtime_case=runtime_case,
            input_mode="image_only",
            decision_policy_version=decision_policy_version,
            tool_health=self.tool_health_summary,
            runtime_store=runtime_store,
        )
        self.last_state = state
        started = time.time()
        try:
            if decision_policy_version != UNIFIED_REACT_POLICY_VERSION:
                raise RuntimeError(
                    "unsupported decision_policy_version="
                    f"{decision_policy_version!r}; only "
                    f"{UNIFIED_REACT_POLICY_VERSION!r} is supported"
                )
            (
                investigation,
                judgment,
                basis,
                final_visual_audit,
            ) = await self._run_react_runtime_policy(
                state,
                image_path=image_path,
                runtime_case=runtime_case,
            )
            state.termination = "success"
            self._sync_image_only_state(state, investigation)
        except Exception as exc:
            state.termination = "error"
            state.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            state.stage_timings["total"] = round(time.time() - started, 2)

        return {
            "image_id": state.image_id,
            "case_id": runtime_case.case_id,
            "image_path": state.image_path,
            "input_mode": state.input_mode,
            "decision_policy_version": state.decision_policy_version,
            "judgment": judgment.model_dump(mode="json"),
            "verdict": judgment.verdict,
            "confidence": judgment.confidence,
            "overall_assessment": judgment.overall_assessment,
            "fact_check_report": (
                judgment.fact_check_report.model_dump(mode="json")
                if judgment.fact_check_report is not None
                else None
            ),
            "evidence_citations": [
                citation.model_dump(mode="json")
                for citation in judgment.evidence_citations
            ],
            "stop_reason": investigation.stop_reason,
            "action_count": investigation.action_count,
            "verdict_basis": (
                basis.model_dump(mode="json")
                if hasattr(basis, "model_dump")
                else dict(basis)
            ),
            "final_visual_audit": final_visual_audit,
            "state": state.to_dict(),
            "termination": state.termination,
            "time_taken": state.stage_timings["total"],
            "token_usage": state.token_usage,
            "total_tool_calls": state.total_tool_calls,
            "total_tool_subcalls": state.total_tool_subcalls,
            "tool_subcalls_by_kind": state.tool_subcalls_by_kind,
            "llm_api_calls": state.llm_api_calls,
            "error": None,
        }

    async def _run_react_runtime_policy(
        self,
        state: VerificationState,
        *,
        image_path: str,
        runtime_case: ImageOnlyRuntimeCase,
    ) -> tuple[
        RuntimeReactState,
        DiscrepancyJudgment,
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]:
        """Run the active one-loop ReAct runtime on one retained Interaction."""

        self._validate_react_tool_configuration()
        investigation = new_unified_react_runtime_state(runtime_case)
        state.investigation_state = investigation
        self._sync_image_only_state(state, investigation)
        started = time.time()
        # Keep one provider-side conversation for the complete ReAct episode.
        # Each outer iteration still creates a fresh StageRunner, but the
        # shared session carries the previous interaction ID and exactly the
        # immediately preceding function result into the next request.
        interaction_session = InteractionSession()
        while not investigation.stop_reason:
            self._check_timeout(started, state)
            if investigation.action_count >= MAX_REACT_ACTIONS:
                investigation.stop_reason = "hard_budget_exhausted"
                break

            tools = build_react_runtime_tools(
                investigation,
                self.all_tools,
                image_path=image_path,
                excluded_tool_names=(
                    set(self.disabled_runtime_tools)
                    | set(
                        self._exhausted_unified_react_tools(state.all_steps)
                    )
                ),
            )
            if not tools:
                investigation.stop_reason = "meaningful_routes_exhausted"
                break

            runner = StageRunner(
                llm=self.llm,
                system_prompt=self._sp(UNIFIED_REACT_SYSTEM_PROMPT),
                prompt_version=UNIFIED_REACT_PROMPT_VERSION,
                tools=tools,
                output_schema=InvestigationSegmentOutput,
                max_rounds=1,
                image_path=image_path,
                stage_name="unified_react",
                runtime_store=state.runtime_store,
                attach_image=(
                    bool(image_path) and self._main_llm_attaches_image()
                ),
                prior_steps=[
                    step
                    for step in state.all_steps
                    if getattr(step, "stage_name", "") == "unified_react"
                ],
                tool_cache=self.tool_cache,
                cacheable_tools=list(self.cacheable_tools),
                tool_call_limits=self.verification_tool_limits,
                min_tool_calls=1,
                should_stop=lambda steps: any(
                    item.action_type == "tool_call" for item in steps
                ),
                max_output_tokens=self._stage_output_tokens(
                    "UNIFIED_REACT",
                    8192,
                ),
                generation_config=self._stage_generation_config(
                    "UNIFIED_REACT"
                ),
                source_access_policy=self.source_access_policy,
                visual_call_validator=lambda tool_name, tool_args: (
                    validate_react_action(
                        investigation,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        source_access_policy=self.source_access_policy,
                        allowed_tool_names=self.enabled_runtime_tools,
                    )
                ),
                max_protocol_corrections=4,
                max_tool_calls_per_turn=1,
                force_tool_each_round=True,
                protocol_exhaustion_boundary=True,
                stop_output_factory=lambda: InvestigationSegmentOutput(
                    segment_summary="One ReAct action completed.",
                    action_completed=True,
                ),
                interaction_session=interaction_session,
                request_timeout_seconds=self.stage_request_timeout_seconds,
                tool_timeout_seconds=self.tool_action_timeout_seconds,
            )
            try:
                parsed, steps = await runner.run(
                    render_react_runtime_context(investigation)
                )
            except Exception as exc:
                self._record_stage_steps(
                    state,
                    list(getattr(exc, "stage_steps", []) or []),
                )
                raise
            action_step = next(
                (
                    item
                    for item in steps
                    if item.action_type == "tool_call"
                ),
                None,
            )
            protocol_boundary = any(
                bool(
                    (item.metadata or {}).get(
                        "protocol_correction_exhaustion_boundary"
                    )
                )
                for item in steps
            )
            if parsed is None:
                self._record_stage_steps(state, steps)
                raise RuntimeError(
                    "ReAct stage returned no structured boundary or action output"
                )
            if action_step is None and protocol_boundary:
                # No tool action was accepted in this StageRunner instance.
                # This is a bounded protocol-repair outcome, not a failed
                # investigation. Hand the accumulated state to the existing
                # terminal Judgment stage.
                investigation.stop_reason = (
                    "protocol_correction_budget_exhausted"
                )
                self._record_stage_steps(state, steps)
                self._sync_image_only_state(state, investigation)
                break
            if action_step is None:
                self._record_stage_steps(state, steps)
                raise RuntimeError(
                    "ReAct did not complete one accepted action"
                )
            call_id = str(
                (action_step.metadata or {}).get("function_call_id", "")
            ).strip()
            record_react_action(
                investigation,
                tool_name=str(action_step.tool_name).strip(),
                tool_args=dict(action_step.tool_args or {}),
                call_id=call_id or f"action-{investigation.action_count}",
            )
            self._record_stage_steps(state, steps)
            self._sync_image_only_state(state, investigation)
            if investigation.action_count >= MAX_REACT_ACTIONS:
                investigation.stop_reason = "hard_budget_exhausted"
                break

        state.stage_timings["unified_react"] = round(time.time() - started, 2)
        if not investigation.stop_reason:
            investigation.stop_reason = "meaningful_routes_exhausted"
        basis = compile_react_judgment_basis(investigation, state.all_steps)
        judgment_started = time.perf_counter()
        try:
            judgment = await self._run_react_judgment(
                state,
                investigation,
                basis,
                image_path=image_path,
                interaction_session=interaction_session,
            )
        finally:
            state.stage_timings["judgment"] = round(
                time.perf_counter() - judgment_started,
                2,
            )
        state.judgment = judgment
        self._sync_image_only_state(state, investigation)
        return investigation, judgment, basis, None

    async def _run_react_judgment(
        self,
        state: VerificationState,
        investigation: RuntimeReactState,
        basis: Dict[str, Any],
        *,
        image_path: str,
        interaction_session: InteractionSession,
    ) -> DiscrepancyJudgment:
        """Run the single terminal binary judgment and report writer."""

        runner = StageRunner(
            llm=self.llm,
            system_prompt=self._sp(UNIFIED_JUDGMENT_SYSTEM_PROMPT),
            prompt_version=UNIFIED_JUDGMENT_PROMPT_VERSION,
            tools=[],
            output_schema=RawHistoryJudgmentOutput,
            max_rounds=1,
            image_path=image_path,
            stage_name="unified_judgment",
            runtime_store=state.runtime_store,
            attach_image=(
                bool(image_path) and self._main_llm_attaches_image()
            ),
            output_validator=lambda parsed, _steps: self._validate_react_judgment(
                parsed,
                basis=basis,
            ),
            max_output_tokens=self._stage_output_tokens(
                "UNIFIED_JUDGMENT",
                8192,
            ),
            generation_config=self._stage_generation_config(
                "UNIFIED_JUDGMENT"
            ),
            request_timeout_seconds=self.stage_request_timeout_seconds,
            interaction_session=interaction_session,
        )
        parsed, steps = await runner.run(
            render_react_judgment_context(investigation, basis)
        )
        self._record_stage_steps(state, steps)
        if parsed is None or parsed.fact_check_report is None:
            raise RuntimeError(
                "ReAct final Judgment did not produce a valid fact-check report"
            )
        valid, validation_error = self._validate_react_judgment(
            parsed,
            basis=basis,
        )
        if not valid:
            raise RuntimeError("ReAct final Judgment failed validation: " + validation_error)

        return DiscrepancyJudgment(
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            policy_rule_id=UNIFIED_REACT_POLICY_VERSION,
            overall_assessment=parsed.overall_assessment,
            fact_check_report=parsed.fact_check_report,
            selected_observation_ids=list(basis.get("observation_ids", []))[:40],
            verdict_observation_ids=list(parsed.verdict_observation_ids),
        )

    def _validate_react_tool_configuration(self) -> None:
        """Validate the image tools required by the active ReAct runtime."""

        if self.vlm_provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ):
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini perception."
            )
        for tool_name in ("perceive_scene", "ocr_with_position"):
            if tool_name not in self.all_tools:
                raise RuntimeError(
                    f"Required image-only bootstrap tool '{tool_name}' is unavailable: "
                    + str(
                        self.tool_health_summary.get(tool_name, {}).get(
                            "error",
                            "not registered",
                        )
                    )
                )

    @staticmethod
    def _validate_react_judgment(
        parsed: RawHistoryJudgmentOutput,
        *,
        basis: Mapping[str, Any],
    ) -> tuple[bool, str]:
        """Keep final reports grounded in retained successful observations."""

        allowed = {
            str(item)
            for item in basis.get("observation_ids", [])
            if str(item).strip()
        }
        cited = [str(item) for item in parsed.verdict_observation_ids]
        unknown = [item for item in cited if item not in allowed]
        if unknown:
            return False, (
                "verdict_observation_ids contains IDs absent from the retained "
                "successful tool observations: "
                + ", ".join(unknown[:6])
            )
        if len(cited) != len(set(cited)):
            return False, "verdict_observation_ids must be unique"
        return True, ""

    def _exhausted_unified_react_tools(
        self,
        steps: Sequence[Any],
    ) -> List[str]:
        """Return hard-exhausted tools before building the next action schema."""

        used: Dict[str, int] = {}
        for step in steps:
            if str(getattr(step, "stage_name", "")).strip() != "unified_react":
                continue
            if str(getattr(step, "action_type", "")).strip() != "tool_call":
                continue
            tool_name = str(getattr(step, "tool_name", "")).strip()
            if tool_name:
                used[tool_name] = used.get(tool_name, 0) + 1
        return [
            tool_name
            for tool_name, limit in self.verification_tool_limits.items()
            if used.get(tool_name, 0) >= int(limit)
        ]

    @staticmethod
    def _sync_image_only_state(
        state: VerificationState,
        investigation: RuntimeReactState,
    ) -> None:
        state.investigation_state = investigation
        if state.runtime_store is not None:
            state.runtime_store.write_snapshot(
                "workspace",
                {
                    "action_count": investigation.action_count,
                    "stop_reason": investigation.stop_reason,
                    "investigation_state": investigation.model_dump(mode="json"),
                    "token_usage": state.token_usage,
                    "total_tool_calls": state.total_tool_calls,
                },
            )

    def _stage_output_tokens(self, stage_name: str, default: int) -> int:
        normalized_stage = stage_name.strip().upper()
        if self.provider in {"qwen_local", "lmdeploy"}:
            env_name = f"QWEN_{normalized_stage}_MAX_OUTPUT_TOKENS"
            provider_default = default
        else:
            env_name = f"GEMINI_{normalized_stage}_MAX_OUTPUT_TOKENS"
            provider_default = default
        value = os.getenv(env_name, str(provider_default)).strip()
        try:
            tokens = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{env_name} must be an integer."
            ) from exc
        if tokens < 1:
            raise ValueError(f"{env_name} must be positive.")
        return tokens

    @staticmethod
    def _stage_thinking_level(stage_name: str) -> str:
        normalized_stage = stage_name.strip().upper()
        default_level = "high" if normalized_stage == "UNIFIED_REACT" else "low"
        fallback = os.getenv("GEMINI_AGENT_THINKING_LEVEL", default_level)
        value = os.getenv(
            f"GEMINI_{normalized_stage}_THINKING_LEVEL",
            fallback,
        ).strip().lower()
        # Keep the existing low-only policy for non-ReAct stages.  The main
        # ReAct loop is the one stage under experiment where Gemini high
        # thinking is useful for selecting and sequencing investigation tools.
        allowed = {"low", "high"} if normalized_stage == "UNIFIED_REACT" else {"low"}
        if value not in allowed:
            raise ValueError(
                f"GEMINI_{normalized_stage}_THINKING_LEVEL must be one of "
                + ", ".join(sorted(allowed))
                + "."
            )
        return value

    def _stage_generation_config(self, stage_name: str) -> Dict[str, Any]:
        """Map one stage's reasoning policy to the active serving protocol."""

        normalized_stage = stage_name.strip().upper()
        if self.provider in {"qwen_local", "lmdeploy"}:
            # Qwen3.5 uses one checkpoint for both thinking and direct-response
            # modes. Both active policy stages retain visible reasoning.
            qwen35 = "qwen3.5" in str(
                getattr(self, "model_name", "")
            ).lower()
            if qwen35:
                thinking_stages = {"UNIFIED_REACT", "UNIFIED_JUDGMENT"}
                default = (
                    "true" if normalized_stage in thinking_stages else "false"
                )
            else:
                default = "true"
            raw = os.getenv(
                f"QWEN_{normalized_stage}_ENABLE_THINKING",
                default,
            ).strip().lower()
            if raw not in {"true", "false"}:
                raise ValueError(
                    f"QWEN_{normalized_stage}_ENABLE_THINKING must be true or false."
                )
            enable_thinking = raw == "true"
            config: Dict[str, Any] = {"enable_thinking": enable_thinking}
            if qwen35:
                # Bound reasoning separately from max_tokens to reserve space for visible JSON.
                config.update(
                    {
                        "temperature": 1.0 if enable_thinking else 0.7,
                        "top_p": 0.95 if enable_thinking else 0.8,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": 1.5,
                        "repetition_penalty": 1.0,
                    }
                )
                if enable_thinking:
                    default_budget = {
                        "UNIFIED_REACT": 2048,
                        "UNIFIED_JUDGMENT": 1024,
                    }.get(normalized_stage, 1024)
                    env_name = f"QWEN_{normalized_stage}_THINKING_TOKEN_BUDGET"
                    raw_budget = os.getenv(env_name, str(default_budget)).strip()
                    try:
                        budget = int(raw_budget)
                    except ValueError as exc:
                        raise ValueError(f"{env_name} must be an integer.") from exc
                    if budget < 1:
                        raise ValueError(f"{env_name} must be positive.")
                    config["thinking_token_budget"] = budget
            sampling_seed = getattr(self, "sampling_seed", None)
            if sampling_seed is not None:
                request_counts = getattr(self, "_sampling_request_counts", {})
                ordinal = request_counts.get(normalized_stage, 0)
                request_counts[normalized_stage] = ordinal + 1
                self._sampling_request_counts = request_counts
                material = (
                    f"{sampling_seed}:{normalized_stage}:{ordinal}"
                ).encode("utf-8")
                config["seed"] = int.from_bytes(
                    hashlib.sha256(material).digest()[:4], "big"
                ) & 0x7FFFFFFF
            return config
        return {
            "thinking_level": self._stage_thinking_level(normalized_stage),
            # Request the provider-generated summary; the hidden chain of
            # thought itself is not available through the API.
            "thinking_summaries": "auto",
        }

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        image_path: str,
        *,
        stage: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        """Execute one tool with cache-miss single-flight protection."""

        if tool_name in self.cacheable_tools and self.tool_cache.enabled:
            cache_tool_args = dict(args)
            properties = self.all_tools[tool_name].parameters.get(
                "properties",
                {},
            )
            if "image_input" in properties and not cache_tool_args.get(
                "image_input"
            ):
                cache_tool_args["image_input"] = image_path
            cache_args = self._build_cache_args(
                tool_name,
                cache_tool_args,
                image_path=image_path,
            )
            async with self.tool_cache.singleflight(tool_name, cache_args):
                return await self._execute_tool_uncached(
                    tool_name,
                    args,
                    image_path,
                    stage=stage,
                )
        return await self._execute_tool_uncached(
            tool_name,
            args,
            image_path,
            stage=stage,
        )

    async def _execute_tool_uncached(
        self,
        tool_name: str,
        args: Dict[str, Any],
        image_path: str,
        *,
        stage: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        tool = self.all_tools[tool_name]
        tool_args = dict(args)
        properties = tool.parameters.get("properties", {})
        if "image_input" in properties and not tool_args.get("image_input"):
            tool_args["image_input"] = image_path
        if hasattr(tool, "image_path") and image_path:
            tool.image_path = image_path

        cache_args = self._build_cache_args(tool_name, tool_args, image_path=image_path)
        started = time.perf_counter()
        runtime_store = current_case_runtime_store()
        if runtime_store is not None:
            recovered = runtime_store.reuse_tool_result(
                stage=stage or "direct_tool",
                tool_name=tool_name,
                tool_args=cache_args,
            )
            if recovered is not None:
                recovered_result = str(recovered["result"])
                _, succeeded = parse_tool_result(recovered_result)
                return recovered_result, {
                    "cache_hit": True,
                    "recovery_cache_hit": True,
                    "recovery_source_attempt_id": recovered.get(
                        "source_attempt_id"
                    ),
                    "recovery_source_memory_id": recovered.get(
                        "source_memory_id"
                    ),
                    "tool_success": succeeded,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    "serialized_size": len(recovered_result),
                }
        if tool_name in self.cacheable_tools:
            cached = self.tool_cache.get(tool_name, cache_args)
            if cached is not None:
                _, succeeded = parse_tool_result(cached)
                return cached, {
                    "cache_hit": True,
                    "tool_success": succeeded,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "serialized_size": len(cached),
                }

        try:
            if hasattr(tool, "call_async"):
                result = await run_tool_with_timeout(
                    tool.call_async(tool_args),
                    timeout_seconds=getattr(
                        self,
                        "tool_action_timeout_seconds",
                        150.0,
                    ),
                )
            else:
                result = await run_tool_with_timeout(
                    asyncio.to_thread(tool.call, tool_args),
                    timeout_seconds=getattr(
                        self,
                        "tool_action_timeout_seconds",
                        150.0,
                    ),
                )
        except asyncio.TimeoutError:
            deadline = getattr(
                self,
                "tool_action_timeout_seconds",
                150.0,
            )
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"ToolActionTimeout: {tool_name} exceeded "
                        f"{deadline:.1f}s"
                    ),
                },
                ensure_ascii=False,
            )
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": "ToolActionTimeout",
                "tool_timeout_seconds": deadline,
                "tool_llm_api_calls": 0,
                "tool_tokens": StageRunner._normalize_tool_tokens(None),
            }
        except Exception as exc:
            runtime_metrics = getattr(exc, "_gemini_runtime_metrics", {})
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            return serialized, {
                "cache_hit": False,
                "tool_success": False,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "serialized_size": len(serialized),
                "tool_exception": type(exc).__name__,
                "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
                "tool_tokens": StageRunner._normalize_tool_tokens(
                    runtime_metrics.get("tokens")
                ),
            }

        runtime_metrics = take_runtime_metrics(result)
        try:
            serialized, succeeded = serialize_tool_result(result)
        except Exception as exc:
            serialized = json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
            succeeded = False
            contract_exception = type(exc).__name__
        else:
            contract_exception = ""
        if succeeded and tool_name in self.cacheable_tools:
            self.tool_cache.put(tool_name, cache_args, serialized)
        return serialized, {
            "cache_hit": False,
            "tool_success": succeeded,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "serialized_size": len(serialized),
            "tool_llm_api_calls": int(runtime_metrics.get("llm_api_calls", 0) or 0),
            "tool_tokens": StageRunner._normalize_tool_tokens(
                runtime_metrics.get("tokens")
            ),
            **({"tool_exception": contract_exception} if contract_exception else {}),
        }

    @staticmethod
    def _build_cache_args(tool_name: str, tool_args: Dict[str, Any], *, image_path: str) -> Dict[str, Any]:
        cache_args = dict(tool_args)
        if tool_name in {"compare_with_reference", "analyze_visual_anomalies"} and image_path:
            cache_args["__image_input__"] = image_path
        if tool_name in {"visit", "crop_and_search"}:
            cache_args["__web_evidence_contract__"] = WEB_EVIDENCE_CONTRACT_VERSION
        return cache_args

    def _record_stage_steps(self, state: VerificationState, steps: Sequence[StageStep]) -> None:
        if not steps:
            return
        state.all_steps.extend(steps)
        state.total_tool_calls += sum(1 for step in steps if step.action_type == "tool_call")
        for step in steps:
            subcalls = step.metadata.get("tool_subcalls", [])
            if isinstance(subcalls, list):
                for subcall in subcalls:
                    if not isinstance(subcall, dict):
                        continue
                    request_count = max(
                        0,
                        int(subcall.get("request_count", 1) or 0),
                    )
                    state.total_tool_subcalls += request_count
                    kind = str(subcall.get("kind", "unknown")).strip() or "unknown"
                    state.tool_subcalls_by_kind[kind] = (
                        state.tool_subcalls_by_kind.get(kind, 0)
                        + request_count
                    )
            tool_tokens = step.metadata.get("tool_tokens", {})
            if not isinstance(tool_tokens, dict):
                tool_tokens = {}
            for name in ("prompt", "completion", "thought"):
                state.token_usage[name] += step.tokens.get(name, 0)
                state.token_usage[name] += int(tool_tokens.get(name, 0) or 0)
            if int(step.metadata.get("tool_llm_api_calls", 0) or 0) > 0:
                step.metadata["total_tokens"] = {
                    name: step.tokens.get(name, 0)
                    + int(tool_tokens.get(name, 0) or 0)
                    for name in ("prompt", "completion", "thought")
                }
        state.llm_api_calls += sum(
            1 for step in steps if step.metadata.get("llm_duration_ms") is not None
        )
        state.llm_api_calls += sum(
            int(step.metadata.get("tool_llm_api_calls", 0) or 0)
            for step in steps
        )

    def _check_timeout(self, started: float, state: VerificationState) -> None:
        if time.time() - started > self.timeout:
            state.termination = "timeout"
            raise TimeoutError(f"Pipeline exceeded timeout of {self.timeout} seconds.")

    @staticmethod
    def _runtime_timeout(env_name: str, default: float) -> float:
        raw = os.getenv(env_name, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            value = default
        return max(5.0, value)
