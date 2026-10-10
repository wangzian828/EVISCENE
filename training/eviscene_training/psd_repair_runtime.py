"""Qwen Chat Completions continuation adapter for EviScene PSD repairs.

The adapter intentionally delegates model calls and tool execution to the
existing runtime StageRunner.  It is not a second tool protocol and it does
not support Gemini interaction IDs as an arbitrary history-replay mechanism.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Sequence

from . import _repo_import  # noqa: F401
from eviscene.orchestrator.source_access import SourceAccessPolicy
from eviscene.orchestrator.investigation_models import (
    InvestigationSegmentOutput,
    RawHistoryJudgmentOutput,
)
from eviscene.orchestrator.llm_backend import LLMResponse
from eviscene.orchestrator.react_runtime import (
    MAX_REACT_ACTIONS,
    REACT_RUNTIME_SCHEMA_VERSION,
    UnifiedReactState,
    build_react_runtime_tools,
    compile_react_judgment_basis,
    record_react_action,
    render_react_judgment_context,
    render_react_runtime_context,
    validate_react_action,
)
from eviscene.orchestrator.stage_runner import StageRunner, StageStep
from eviscene.orchestrator.tool_cache import ToolResultCache
from eviscene.orchestrator.unified_prompts import UNIFIED_JUDGMENT_SYSTEM_PROMPT, UNIFIED_REACT_SYSTEM_PROMPT
from eviscene.tools.base import BaseTool

from .psd_repair import (
    FailureSite,
    HintProposal,
    PSDModelRoles,
    build_proposer_prompt,
    build_student_messages,
    build_teacher_messages,
    build_hint_proposal,
    parse_proposer_response,
    restore_failure_site_archive,
)
from .psd_repair_verifier import build_complete_hinted_episode_trace


@dataclass(frozen=True)
class ContinuationResult:
    teacher_steps: List[StageStep]
    student_steps: List[StageStep]
    teacher_history: List[Dict[str, Any]]
    student_history: List[Dict[str, Any]]
    hint: str
    teacher_complete: bool = False
    student_complete: bool = False
    stop_reason: str = ""
    teacher_episode_trace: Dict[str, Any] | None = None
    local_targets: List[Dict[str, Any]] = field(default_factory=list)


class QwenContinuationAdapter:
    """Run a hinted suffix and a no-hint continuation from the same raw state."""

    def __init__(
        self,
        *,
        policy_llm: Any,
        hint_constructor_llm: Any,
        model_roles: PSDModelRoles,
        tools: Sequence[BaseTool],
        image_path: str,
        runtime_store: Any = None,
        tool_cache: ToolResultCache | None = None,
        cacheable_tools: Sequence[str] = (),
        source_access_policy: SourceAccessPolicy | None = None,
        tool_call_limits: Mapping[str, int] | None = None,
        max_output_tokens: int = 8192,
        judgment_max_output_tokens: int | None = None,
        generation_config: Mapping[str, Any] | None = None,
        judgment_generation_config: Mapping[str, Any] | None = None,
        judgment_system_prompt: str = "",
        react_system_prompt: str = "",
        request_timeout_seconds: float | None = None,
        tool_timeout_seconds: float | None = None,
        require_runtime_archive: bool = True,
        source_runtime_store_path: str = "",
        policy_topk: int = 20,
        hint_constructor_thinking_level: str = "low",
    ) -> None:
        from .psd_capture_semantics import install_capture_semantics
        install_capture_semantics()
        provider = str(getattr(policy_llm, "provider", "")).strip().lower()
        wire_api = str(getattr(policy_llm, "wire_api", "")).strip().lower()
        if provider not in {"qwen_local", "lmdeploy"} or wire_api != "chat_completions":
            raise ValueError(
                "PSD repair continuation requires a Qwen-compatible Chat Completions backend"
            )
        hint_provider = str(
            getattr(hint_constructor_llm, "provider", "")
        ).strip().lower()
        hint_model = str(
            getattr(hint_constructor_llm, "model_name", "")
        ).strip()
        policy_model = str(getattr(policy_llm, "model_name", "")).strip()
        if (
            provider != model_roles.frozen_self_teacher_provider.strip().lower()
            or policy_model != model_roles.frozen_self_teacher_model.strip()
        ):
            raise ValueError("policy backend does not match frozen self-teacher role")
        if (
            hint_provider != model_roles.hint_constructor_provider.strip().lower()
            or hint_model != model_roles.hint_constructor_model.strip()
        ):
            raise ValueError("hint backend does not match hint-constructor role")
        self.policy_llm = policy_llm
        self.hint_constructor_llm = hint_constructor_llm
        self.model_roles = model_roles
        self.tools = list(tools)
        self.tools_by_name = {
            str(tool.name).strip(): tool
            for tool in self.tools
            if str(tool.name).strip()
        }
        self.image_path = image_path
        self.runtime_store = runtime_store
        self.tool_cache = tool_cache
        self.cacheable_tools = list(cacheable_tools)
        self.source_access_policy = source_access_policy or SourceAccessPolicy()
        self.tool_call_limits = dict(tool_call_limits or {})
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.judgment_max_output_tokens = max(1, int(
            judgment_max_output_tokens if judgment_max_output_tokens is not None else max_output_tokens))
        self.generation_config = dict(generation_config or {})
        self.judgment_generation_config = dict(
            judgment_generation_config or self.generation_config
        )
        self.judgment_system_prompt = str(
            judgment_system_prompt or UNIFIED_JUDGMENT_SYSTEM_PROMPT
        )
        self.react_system_prompt = str(react_system_prompt or UNIFIED_REACT_SYSTEM_PROMPT)
        self.request_timeout_seconds = request_timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.require_runtime_archive = bool(require_runtime_archive)
        self.source_runtime_store_path = str(source_runtime_store_path or "").strip()
        self.policy_topk = max(1, min(100, int(policy_topk)))
        self.hint_constructor_thinking_level = str(
            hint_constructor_thinking_level or "low"
        ).strip().lower()
        if self.hint_constructor_thinking_level not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ValueError("unsupported hint-constructor thinking level")

    def _site(self, site: FailureSite) -> FailureSite:
        if (
            self.source_runtime_store_path
            and site.runtime_store_path != self.source_runtime_store_path
        ):
            site = replace(
                site,
                runtime_store_path=self.source_runtime_store_path,
            )
        return restore_failure_site_archive(
            site,
            require=self.require_runtime_archive,
        )

    async def _guard_policy(self):
        callback = getattr(self, "before_policy_call", None)
        if callback is not None:
            await callback()

    def _runner(
        self,
        *,
        site: FailureSite,
        history: Sequence[Mapping[str, Any]],
        include_hint_as_pending_user: bool,
        stage_name: str,
        prior_steps: Sequence[StageStep] = (),
        tools: Sequence[BaseTool] | None = None,
        visual_call_validator: Any = None,
    ) -> StageRunner:
        return StageRunner(
            llm=self.policy_llm,
            # Archived system_instruction is ALREADY rendered by StageRunner.
            # Reusing it as the base prompt duplicates schema/availability blocks.
            # Use the same unrendered prompt as the frozen Orchestrator instead.
            system_prompt=self.react_system_prompt,
            tools=list(tools if tools is not None else self.tools),
            output_schema=InvestigationSegmentOutput,
            max_rounds=1,
            image_path=self.image_path,
            stage_name=stage_name,
            runtime_store=self.runtime_store,
            attach_image=False,
            prior_steps=list(prior_steps),
            tool_cache=self.tool_cache,
            cacheable_tools=self.cacheable_tools,
            tool_call_limits=self.tool_call_limits,
            min_tool_calls=1,
            should_stop=lambda steps: any(
                step.action_type == "tool_call" for step in steps
            ),
            force_tool_each_round=True,
            stop_output_factory=lambda: InvestigationSegmentOutput(
                segment_summary="PSD repair continuation action completed.",
                action_completed=True,
            ),
            protocol_exhaustion_boundary=True,
            max_tool_calls_per_turn=1,
            max_protocol_corrections=4,
            max_output_tokens=self.max_output_tokens,
            generation_config=self.generation_config,
            source_access_policy=self.source_access_policy,
            visual_call_validator=visual_call_validator,
            request_timeout_seconds=self.request_timeout_seconds,
            tool_timeout_seconds=self.tool_timeout_seconds,
            capture_policy_tokens=True,
            policy_topk=self.policy_topk,
            native_history=self._native_history(history, _text(site.policy_input.get("system_instruction"))),
            native_system_instruction=_text(site.policy_input.get("system_instruction")),
            native_history_includes_pending_user=include_hint_as_pending_user,
        )

    @staticmethod
    def _native_history(history, system_instruction):
        """The request archive includes its system message; StageRunner adds it.

        Remove only the identical leading transport copy. Never edit the source
        user/assistant/tool prefix, and reject any non-leading system message.
        """
        rows = [dict(item) for item in history]
        if rows and rows[0].get("role") == "system":
            if rows[0].get("content") != system_instruction:
                raise ValueError("PSD archived system instruction differs from replay")
            rows = rows[1:]
        if any(row.get("role") == "system" for row in rows):
            raise ValueError("PSD native history contains an interior system message")
        return rows

    @staticmethod
    def _remove_local_hint(history, *, unhinted_prefix, hint):
        """Remove exactly the injected message, never a matching observed string.

        The archived teacher request retains the hint for scoring. Only the
        native history used for later decisions is changed, as in upstream's
        hint-free fold. Repeated text inside actual observations is preserved.
        """
        rows = copy.deepcopy(list(history))
        index = len(unhinted_prefix)
        if rows[:index] != list(unhinted_prefix):
            raise ValueError("PSD continuation changed the pre-hint history")
        if index >= len(rows) or rows[index] != {"role": "user", "content": hint}:
            raise ValueError("PSD local hint is absent from its bound message position")
        del rows[index]
        return rows

    @staticmethod
    def _remaining_actions(runtime_state, requested):
        remaining = max(0, MAX_REACT_ACTIONS - runtime_state.action_count)
        if requested is None:
            return remaining
        if type(requested) is not int or requested < 1:
            raise ValueError("max_suffix_actions must be positive or omitted")
        return min(requested, remaining)

    @staticmethod
    def _stage_step_from_row(row: Mapping[str, Any]) -> StageStep:
        return StageStep(
            round=int(row.get("round", 0) or 0),
            stage_name=_text(row.get("stage")),
            thought=_text(row.get("thought")),
            action_type=_text(row.get("action_type")),
            tool_name=_text(row.get("tool_name")),
            tool_args=dict(row.get("tool_args") or {}),
            tool_result=_text(row.get("tool_result")),
            output=(
                dict(row["output"])
                if isinstance(row.get("output"), Mapping)
                else None
            ),
            tokens=dict(row.get("tokens") or {}),
            metadata=copy.deepcopy(dict(row.get("metadata") or {})),
        )

    def _initial_runtime_state(
        self,
        *,
        base_trace: Mapping[str, Any],
        failure_site: FailureSite,
    ) -> tuple[UnifiedReactState, list[StageStep]]:
        source_index = failure_site.source_step_index
        state = base_trace.get("state")
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or not isinstance(state, Mapping)
        ):
            raise ValueError("PSD continuation requires an indexed source state")
        rows = state.get("all_steps")
        if not isinstance(rows, list) or not (0 <= source_index < len(rows)):
            raise ValueError("PSD continuation source step index is out of range")
        runtime_case = state.get("runtime_case")
        runtime_case = runtime_case if isinstance(runtime_case, Mapping) else {}
        source_investigation = state.get("investigation_state")
        source_investigation = (
            source_investigation
            if isinstance(source_investigation, Mapping)
            else {}
        )
        case_id = _text(
            runtime_case.get("case_id") or source_investigation.get("case_id")
        )
        image_sha256 = _text(
            runtime_case.get("image_sha256")
            or source_investigation.get("image_sha256")
        )
        if not case_id or len(image_sha256) != 64:
            raise ValueError("PSD continuation lacks a valid runtime case binding")
        runtime_state = UnifiedReactState(
            schema_version=REACT_RUNTIME_SCHEMA_VERSION,
            case_id=case_id,
            image_sha256=image_sha256,
            objective=_text(source_investigation.get("objective"))
            or UnifiedReactState.model_fields["objective"].default,
        )
        prefix = [
            self._stage_step_from_row(row)
            for row in rows[:source_index]
            if isinstance(row, Mapping)
        ]
        for step in prefix:
            if step.stage_name == "unified_react" and step.action_type == "tool_call":
                record_react_action(
                    runtime_state,
                    tool_name=step.tool_name,
                    tool_args=step.tool_args,
                    call_id=_text(step.metadata.get("function_call_id")) or "missing",
                )
        return runtime_state, prefix

    def _excluded_tools(self, steps: Sequence[StageStep]) -> list[str]:
        counts: dict[str, int] = {}
        for step in steps:
            if step.action_type == "tool_call" and step.tool_name:
                counts[step.tool_name] = counts.get(step.tool_name, 0) + 1
        return [
            name
            for name, limit in self.tool_call_limits.items()
            if counts.get(name, 0) >= int(limit)
        ]

    @staticmethod
    def _judgment_basis(runtime_state, steps):
        # The frozen locator accepts unified_react only. PSD's archive stage
        # names are diagnostic names, not a reason to drop genuine observations.
        # Normalize copies for this reader; preserve original steps/IDs/errors.
        normalized = [replace(step, stage_name="unified_react")
                      if step.stage_name == "psd_teacher_repair" else step for step in steps]
        return compile_react_judgment_basis(runtime_state, normalized)

    def _judgment_runner(
        self,
        *,
        history: Sequence[Mapping[str, Any]],
        basis: Mapping[str, Any],
        include_pending_user: bool,
        system_instruction: str | None = None,
    ) -> StageRunner:
        allowed = {
            _text(item) for item in basis.get("observation_ids", []) if _text(item)
        }

        def validate(parsed: RawHistoryJudgmentOutput, _steps: list[StageStep]):
            cited = [_text(item) for item in parsed.verdict_observation_ids]
            unknown = [item for item in cited if item not in allowed]
            if unknown:
                return False, "unknown verdict observation IDs: " + ",".join(unknown)
            if len(cited) != len(set(cited)):
                return False, "verdict observation IDs must be unique"
            return True, ""

        # system_instruction, when supplied, is the archived rendered copy used
        # to strip the leading transport system message, not a base prompt.
        instruction = self.judgment_system_prompt
        return StageRunner(
            llm=self.policy_llm,
            system_prompt=instruction,
            tools=[],
            output_schema=RawHistoryJudgmentOutput,
            max_rounds=1,
            image_path=self.image_path,
            stage_name="psd_teacher_judgment",
            runtime_store=self.runtime_store,
            attach_image=False,
            output_validator=validate,
            max_output_tokens=self.judgment_max_output_tokens,
            generation_config=self.judgment_generation_config,
            source_access_policy=self.source_access_policy,
            request_timeout_seconds=self.request_timeout_seconds,
            capture_policy_tokens=True,
            policy_topk=self.policy_topk,
            native_history=self._native_history(history, system_instruction or instruction),
            native_system_instruction=instruction,
            native_history_includes_pending_user=include_pending_user,
        )

    async def propose_hints(
        self,
        *,
        failure_site: FailureSite,
        public_trace_context: Mapping[str, Any],
        private_context: Mapping[str, Any] | None = None,
        hint_count: int = 4,
        hint_level: int = 1,
    ) -> list[HintProposal]:
        """Ask the privileged hint constructor for audited procedural hints."""

        failure_site = self._site(failure_site)
        prompt = build_proposer_prompt(
            failure_site=failure_site,
            public_trace_context=public_trace_context,
            private_context=private_context,
            hint_count=hint_count,
        )
        from .psd_media import proposer_image_inputs
        _, native_images = proposer_image_inputs({"policy_input": failure_site.policy_input,
            "trace_context": public_trace_context})
        native_images.extend(getattr(self, "search_review_images", []))
        messages = [
            {
                "role": "system",
                "content": (
                    "Return JSON only. You are the privileged repair proposer; "
                    "the generated hints must not reveal the answer or exact action."
                ),
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}, *native_images]
             if native_images else prompt},
        ]
        response = await self._call_proposer(messages)
        try:
            payload = json.loads(response.text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("privileged proposer returned non-JSON output") from exc
        proposals: list[HintProposal] = []
        proposal_audits = []
        candidate_id = failure_site.step_id
        for text in parse_proposer_response(payload, limit=hint_count):
            try:
                proposals.append(
                    build_hint_proposal(
                        text=text,
                        level=hint_level,
                        provider=str(
                            getattr(
                                self.hint_constructor_llm,
                                "provider",
                                "",
                            )
                        ),
                        model=str(
                            getattr(self.hint_constructor_llm, "model_name", "")
                        ),
                        candidate_id=candidate_id,
                        public_failure_context={
                            "failure_site": failure_site.public_record(),
                            "trace_context": dict(public_trace_context),
                        },
                        private_context=private_context,
                    )
                )
                proposal_audits.append({"text": text, "passed": True, "audit": proposals[-1].audit})
            except ValueError as exc:
                proposal_audits.append({"text": text, "passed": False, "reason": str(exc)})
                continue
        if getattr(self, "proposer_cache_path", None):
            from .io import write_json
            write_json(self.proposer_cache_path.with_name("proposer-hint-audits.json"), {"proposals": proposal_audits})
        return proposals

    async def _call_proposer(
        self,
        messages: list[dict[str, Any]],
    ) -> Any:
        """Call the privileged proposer while preserving an auditable request."""

        from .psd_repair import _sha
        from .psd_repair_storage import load_bound, save_bound
        cache = getattr(self, "proposer_cache_path", None)
        identity = {"messages_sha256": _sha(messages), "model": self.hint_constructor_llm.model_name,
            "provider": self.hint_constructor_llm.provider, "wire_api": self.hint_constructor_llm.wire_api,
            "thinking_level": self.hint_constructor_thinking_level, "max_tokens": min(self.max_output_tokens, 2048)}
        if cache and cache.exists():
            return LLMResponse(**load_bound(cache, identity=identity))
        request_id = ""
        hint_schema = {
            "type": "object",
            "properties": {
                "hints": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["hints"],
            "additionalProperties": False,
        }
        is_gemini = str(
            getattr(self.hint_constructor_llm, "provider", "")
        ).strip().lower() == "gemini"
        response_format = (
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": hint_schema,
            }
            if is_gemini
            else {"type": "json_object"}
        )
        generation_config = (
            {"thinking_level": self.hint_constructor_thinking_level}
            if is_gemini
            else {"enable_thinking": False}
        )
        if self.runtime_store is not None:
            request_id = self.runtime_store.context_ledger.begin_request(
                stage="psd_proposer",
                lifecycle_kind="privileged_proposer",
                system_instruction=messages[0].get("content", "") if messages else "",
                input_payload=messages[1:] if len(messages) > 1 else messages,
                tools=[],
                response_format=response_format,
                generation_config=generation_config,
                max_output_tokens=min(self.max_output_tokens, 2048),
                model=str(
                    getattr(self.hint_constructor_llm, "model_name", "")
                ),
                prompt_version="ifv-psd-repair-proposer-v4-public-feedback",
            )
        started = time.perf_counter()
        try:
            if is_gemini:
                from eviscene.integrations.gemini import (
                    extract_text,
                    messages_to_input,
                    validate_interaction_response,
                )

                interaction = await self.hint_constructor_llm.create_interaction(
                    input_payload=messages_to_input(messages[1:]),
                    system_instruction=messages[0].get("content", ""),
                    response_format=response_format,
                    store=True,
                    max_tokens=min(self.max_output_tokens, 2048),
                    temperature=0.0,
                    generation_config=generation_config,
                )
                _, status = validate_interaction_response(interaction)
                if status != "completed":
                    raise RuntimeError(
                        "Gemini PSD hint constructor requires status=completed, "
                        f"received status={status}"
                    )
                usage = interaction.get("usage")
                usage = usage if isinstance(usage, Mapping) else {}
                response = LLMResponse(
                    text=extract_text(interaction),
                    prompt_tokens=int(
                        usage.get(
                            "total_input_tokens",
                            0,
                        )
                        or 0
                    ),
                    completion_tokens=int(
                        usage.get(
                            "total_output_tokens",
                            0,
                        )
                        or 0
                    ),
                    raw=dict(interaction),
                )
            else:
                response = await self.hint_constructor_llm.get_response(
                    messages,
                    max_tokens=min(self.max_output_tokens, 2048),
                    response_format=response_format,
                    generation_config=generation_config,
                )
        except Exception as exc:
            if request_id:
                self.runtime_store.context_ledger.complete_request(
                    request_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    response_metadata={
                        "psd_proposer": True,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
            raise
        if cache:
            save_bound(cache, identity=identity, payload={"text": response.text,
                "raw": response.raw, "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens})
        if request_id:
            raw = response.raw if isinstance(getattr(response, "raw", None), dict) else {}
            usage = raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {
                "input_tokens": getattr(response, "prompt_tokens", 0),
                "output_tokens": getattr(response, "completion_tokens", 0),
            }
            self.runtime_store.context_ledger.complete_request(
                request_id,
                usage=usage,
                status="completed",
                interaction_id=_text(raw.get("id")) or None,
                response_metadata={
                    "psd_proposer": True,
                    "provider": str(
                        getattr(self.hint_constructor_llm, "provider", "")
                    ),
                    "model": str(
                        getattr(self.hint_constructor_llm, "model_name", "")
                    ),
                    "wire_api": str(
                        getattr(self.hint_constructor_llm, "wire_api", "")
                    ),
                    "interaction_id": _text(raw.get("id")),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "response_content_chars": len(str(getattr(response, "text", "") or "")),
                },
            )
        return response

    async def run_hinted_suffix(
        self,
        *,
        failure_site: FailureSite,
        hint: HintProposal,
        prior_steps: Sequence[StageStep] = (),
    ) -> ContinuationResult:
        failure_site = self._site(failure_site)
        teacher_history = build_teacher_messages(failure_site, hint)
        teacher_runner = self._runner(
            site=failure_site,
            history=teacher_history,
            include_hint_as_pending_user=True,
            stage_name="psd_teacher_repair",
            prior_steps=prior_steps,
        )
        await self._guard_policy()
        _, teacher_steps = await teacher_runner.run("")
        if not teacher_runner.last_native_history:
            raise RuntimeError("teacher continuation did not retain native history")
        # Student and teacher must be sampled at the identical failed state.
        # Removing the hint from the teacher's post-action history would instead
        # expose the teacher's repaired tool call to the student.
        student_history = build_student_messages(failure_site)
        student_runner = self._runner(
            site=failure_site,
            history=student_history,
            include_hint_as_pending_user=True,
            stage_name="psd_student_continuation",
            prior_steps=list(prior_steps),
        )
        await self._guard_policy()
        _, student_steps = await student_runner.run("")
        return ContinuationResult(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_history=teacher_runner.last_native_history,
            student_history=student_runner.last_native_history,
            hint=hint.text,
        )

    async def run_hinted_episode(
        self,
        *,
        failure_site: FailureSite,
        hint: HintProposal | None,
        base_trace: Mapping[str, Any],
        prior_steps: Sequence[StageStep] = (),
        max_suffix_actions: int | None = None,
        run_student_diagnostic: bool = False,
        hints_by_action: Mapping[int, HintProposal] | None = None,
        capture_local_target: Any = None,
    ) -> ContinuationResult:
        """Generate one complete hinted teacher episode from the failed state.

        Re-entering ``StageRunner`` preserves native provider history while
        keeping the runtime validators, tool budgets, archive and media path in
        the loop. The original no-hint rollout already proves student failure;
        an extra no-hint action is optional diagnostics only.
        """

        site = self._site(failure_site)
        student_history = build_student_messages(site)
        teacher_history = build_teacher_messages(site, hint) if hint is not None else copy.deepcopy(student_history)
        runtime_state, source_prefix = self._initial_runtime_state(
            base_trace=base_trace,
            failure_site=site,
        )
        action_budget = self._remaining_actions(runtime_state, max_suffix_actions)
        hint_free_prefix = self._native_history(student_history, _text(site.policy_input.get("system_instruction")))
        # EviScene has one user task, so slate positions are native decision ordinals,
        # not fictitious extra user turns. The terminal judgment has its own key.
        initial_position = MAX_REACT_ACTIONS if site.stage == "unified_judgment" else runtime_state.action_count
        slate = dict(hints_by_action or {})
        if any(type(k) is not int or not initial_position <= k <= MAX_REACT_ACTIONS for k in slate):
            raise ValueError("PSD slate position is outside the repair episode")
        if hint is not None and initial_position in slate and slate[initial_position] != hint:
            raise ValueError("PSD initial slate hint differs from the selected repair hint")
        if hint is not None:
            slate[initial_position] = hint
        elif initial_position in slate:
            teacher_history = build_teacher_messages(site, slate[initial_position])
        # An explicitly supplied empty slate is the upstream plain-retry path.
        # It produces no intervention target, even if the full episode passes.
        if not slate and hints_by_action is None:
            raise ValueError("PSD repair requires at least one explicit intervention")
        if hints_by_action is not None and capture_local_target is None:
            raise ValueError("PSD multi-position repair requires exact local target capture")
        local_targets = []
        used_positions = set()
        teacher_steps: list[StageStep] = []
        student_steps: list[StageStep] = []
        teacher_complete = False
        student_complete = False

        if site.stage != "unified_judgment":
            for action_index in range(action_budget):
                position = runtime_state.action_count
                local_hint = slate.get(position)
                unhinted_prefix = hint_free_prefix if action_index == 0 else None
                if action_index > 0 and local_hint is not None:
                    unhinted_prefix = [*teacher_history,
                        {"role": "user", "content": render_react_runtime_context(runtime_state)}]
                    teacher_history = [*unhinted_prefix, {"role": "user", "content": local_hint.text}]
                active_tools = build_react_runtime_tools(
                    runtime_state,
                    self.tools_by_name,
                    image_path=self.image_path,
                    excluded_tool_names=self._excluded_tools(
                        [*source_prefix, *prior_steps, *teacher_steps]
                    ),
                )
                if not active_tools:
                    runtime_state.stop_reason = "meaningful_routes_exhausted"
                    break
                runner = self._runner(
                    site=site,
                    history=teacher_history,
                    include_hint_as_pending_user=action_index == 0 or local_hint is not None,
                    stage_name="psd_teacher_repair",
                    prior_steps=[*source_prefix, *prior_steps, *teacher_steps],
                    tools=active_tools,
                    visual_call_validator=lambda tool_name, tool_args: (
                        validate_react_action(
                            runtime_state,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            source_access_policy=self.source_access_policy,
                        )
                    ),
                )
                await self._guard_policy()
                _, steps = await runner.run(
                    "" if action_index == 0 or local_hint is not None else render_react_runtime_context(runtime_state)
                )
                teacher_steps.extend(steps)
                if not runner.last_native_history:
                    raise RuntimeError(
                        "teacher continuation did not retain native history"
                    )
                teacher_history = runner.last_native_history
                if local_hint is not None:
                    teacher_history = self._remove_local_hint(
                        teacher_history, unhinted_prefix=unhinted_prefix, hint=local_hint.text)
                    used_positions.add(position)
                    if capture_local_target is not None:
                        local_targets.append(await capture_local_target(position=position,
                            hint=local_hint, steps=steps, unhinted_prefix=unhinted_prefix,
                            runtime_store=self.runtime_store,
                            system_instruction=_text(site.policy_input.get("system_instruction"))))
                action = next(
                    (step for step in steps if step.action_type == "tool_call"),
                    None,
                )
                if action is None:
                    runtime_state.stop_reason = "protocol_correction_budget_exhausted"
                    break
                record_react_action(
                    runtime_state,
                    tool_name=action.tool_name,
                    tool_args=action.tool_args,
                    call_id=_text(action.metadata.get("function_call_id")) or "missing",
                )
                if runtime_state.stop_reason:
                    break
            if not runtime_state.stop_reason:
                runtime_state.stop_reason = (
                    "hard_budget_exhausted" if runtime_state.action_count >= MAX_REACT_ACTIONS
                    else "psd_suffix_action_limit")

        basis = self._judgment_basis(
            runtime_state,
            [*source_prefix, *teacher_steps],
        )
        judgment_hint = slate.get(MAX_REACT_ACTIONS)
        judgment_prefix = hint_free_prefix if site.stage == "unified_judgment" else None
        if judgment_hint is not None and site.stage != "unified_judgment":
            judgment_prefix = [*teacher_history,
                {"role": "user", "content": render_react_judgment_context(runtime_state, basis)}]
            teacher_history = [*judgment_prefix, {"role": "user", "content": judgment_hint.text}]
        judgment_runner = self._judgment_runner(
            history=teacher_history,
            basis=basis,
            include_pending_user=judgment_hint is not None,
            system_instruction=_text(site.policy_input.get("system_instruction")) if site.stage == "unified_judgment" else None,
        )
        await self._guard_policy()
        parsed_judgment, judgment_steps = await judgment_runner.run(
            ""
            if judgment_hint is not None
            else render_react_judgment_context(runtime_state, basis)
        )
        if parsed_judgment is None or not judgment_runner.last_native_history:
            raise RuntimeError("teacher continuation did not reach final Judgment")
        teacher_history = judgment_runner.last_native_history
        if judgment_hint is not None:
            teacher_history = self._remove_local_hint(
                teacher_history, unhinted_prefix=judgment_prefix, hint=judgment_hint.text)
            used_positions.add(MAX_REACT_ACTIONS)
            if capture_local_target is not None:
                local_targets.append(await capture_local_target(position=MAX_REACT_ACTIONS,
                    hint=judgment_hint, steps=judgment_steps, unhinted_prefix=judgment_prefix,
                    runtime_store=self.runtime_store,
                    system_instruction=(_text(site.policy_input.get("system_instruction"))
                        if site.stage == "unified_judgment" else self.judgment_system_prompt)))
        teacher_steps.extend(judgment_steps)
        teacher_complete = True

        if run_student_diagnostic and site.stage != "unified_judgment":
            student_state, _ = self._initial_runtime_state(
                base_trace=base_trace,
                failure_site=site,
            )
            student_tools = build_react_runtime_tools(
                student_state,
                self.tools_by_name,
                image_path=self.image_path,
                excluded_tool_names=self._excluded_tools(source_prefix),
            )
            student_runner = self._runner(
                site=site,
                history=student_history,
                include_hint_as_pending_user=True,
                stage_name="psd_student_continuation",
                prior_steps=[*source_prefix, *prior_steps],
                tools=student_tools,
                visual_call_validator=lambda tool_name, tool_args: (
                    validate_react_action(
                        student_state,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        source_access_policy=self.source_access_policy,
                    )
                ),
            )
            await self._guard_policy()
            _, student_steps = await student_runner.run("")
            student_history = student_runner.last_native_history
            student_complete = False

        teacher_episode_trace = build_complete_hinted_episode_trace(
            base_trace,
            failure_site=site,
            teacher_steps=teacher_steps,
            stop_reason=runtime_state.stop_reason,
        )
        if hints_by_action is not None:
            teacher_episode_trace["psd_repair"]["slate"] = {
                "schema_version": "ifv-psd-decision-slate-v1",
                "positions": sorted(slate), "used_positions": sorted(used_positions),
                "unused_positions": sorted(set(slate) - used_positions),
                "hint_free_history_after_each_decision": True,
                "local_target_count": len(local_targets)}
        if self.runtime_store is not None:
            teacher_episode_trace["state"]["runtime_store"] = dict(self.runtime_store.descriptor)
            teacher_episode_trace["psd_repair"]["source_runtime_store_path"] = site.runtime_store_path

        return ContinuationResult(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_history=teacher_history,
            student_history=student_history,
            hint=hint.text if hint is not None else "",
            teacher_complete=teacher_complete,
            student_complete=student_complete,
            stop_reason=(
                "teacher_episode_complete"
                if teacher_complete
                else "bounded_suffix_exhausted"
            ),
            teacher_episode_trace=teacher_episode_trace,
            local_targets=local_targets,
        )


def _text(value: Any) -> str:
    return str(value or "").strip()
