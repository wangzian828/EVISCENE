"""Explicit provider profiles for teacher and student runtime isolation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    default_model: Optional[str]
    model_env: Optional[str]
    vlm_provider: str
    default_vlm_model: Optional[str] = None
    vlm_model_env: Optional[str] = None
    llm_wire_api: str = "chat_completions"
    vlm_wire_api: str = "chat_completions"
    default_base_url: Optional[str] = None
    base_url_env: Optional[str] = None


@dataclass(frozen=True)
class ResolvedProviderSettings:
    profile_id: Optional[str]
    provider: str
    model_name: str
    vlm_provider: str
    vlm_model: str
    llm_wire_api: Optional[str]
    vlm_wire_api: Optional[str]
    base_url: Optional[str]
    vlm_base_url: Optional[str]


PROVIDER_PROFILES = {
    "teacher-gemini": ProviderProfile(
        profile_id="teacher-gemini",
        provider="gemini",
        default_model="gemini-3.7-flash",
        model_env=None,
        vlm_provider="gemini",
        default_vlm_model="gemini-3.7-flash",
        llm_wire_api="interactions",
        vlm_wire_api="interactions",
    ),
    "teacher-gemini36": ProviderProfile(
        profile_id="teacher-gemini36",
        provider="gemini",
        default_model="gemini-3.6-flash",
        model_env=None,
        vlm_provider="gemini",
        default_vlm_model="gemini-3.6-flash",
        llm_wire_api="interactions",
        vlm_wire_api="interactions",
    ),
    "teacher-qwen-server": ProviderProfile(
        profile_id="teacher-qwen-server",
        provider="qwen_local",
        default_model=None,
        model_env="QWEN_TEACHER_MODEL",
        vlm_provider="qwen_local",
        default_vlm_model=None,
        vlm_model_env="QWEN_TEACHER_VISION_MODEL",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
        base_url_env="QWEN_TEACHER_BASE_URL",
    ),
    "student-qwen3.5-local": ProviderProfile(
        profile_id="student-qwen3.5-local",
        provider="qwen_local",
        default_model="eviscene",
        model_env="QWEN35_LOCAL_MODEL",
        vlm_provider="qwen_local",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
        default_base_url="http://127.0.0.1:8901/v1",
        base_url_env="QWEN35_LOCAL_BASE_URL",
    ),
    "student-qwen3.5-local-replica-b": ProviderProfile(
        profile_id="student-qwen3.5-local-replica-b",
        provider="qwen_local",
        default_model="eviscene-replica-b",
        model_env="QWEN35_REPLICA_B_LOCAL_MODEL",
        vlm_provider="qwen_local",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
        default_base_url="http://127.0.0.1:8902/v1",
        base_url_env="QWEN35_REPLICA_B_LOCAL_BASE_URL",
    ),
    "student-qwen-api": ProviderProfile(
        profile_id="student-qwen-api",
        provider="qwen",
        default_model=None,
        model_env="QWEN_API_MODEL",
        vlm_provider="qwen",
        vlm_model_env="QWEN_API_VISION_MODEL",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
    ),
}
PROFILE_IDS = tuple(PROVIDER_PROFILES)


def _clean(value: Optional[str]) -> Optional[str]:
    rendered = str(value or "").strip()
    return rendered or None


def _profile_model(
    profile: ProviderProfile,
    environ: Mapping[str, str],
    *,
    vision: bool = False,
    fallback: Optional[str] = None,
) -> str:
    env_name = profile.vlm_model_env if vision else profile.model_env
    default = profile.default_vlm_model if vision else profile.default_model
    value = _clean(environ.get(env_name, "")) if env_name else None
    value = value or _clean(default) or _clean(fallback)
    if not value:
        role = "vision model" if vision else "model"
        requirement = env_name or "an explicit profile model"
        raise ValueError(
            f"provider profile {profile.profile_id!r} requires {role} via {requirement}"
        )
    return value


def resolve_provider_settings(
    *,
    profile_id: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    vlm_provider: Optional[str] = None,
    vlm_model: Optional[str] = None,
    llm_wire_api: Optional[str] = None,
    vlm_wire_api: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ResolvedProviderSettings:
    """Resolve one profile or the legacy loose settings, never both."""

    normalized_profile = _clean(profile_id)
    if not normalized_profile and all(
        value is None
        for value in (
            provider,
            model_name,
            vlm_provider,
            vlm_model,
            llm_wire_api,
            vlm_wire_api,
        )
    ):
        normalized_profile = "student-qwen3.5-local"
    if normalized_profile:
        conflicts = [
            name
            for name, value in {
                "provider": provider,
                "model_name": model_name,
                "vlm_provider": vlm_provider,
                "vlm_model": vlm_model,
                "llm_wire_api": llm_wire_api,
                "vlm_wire_api": vlm_wire_api,
            }.items()
            if value is not None
        ]
        if conflicts:
            raise ValueError(
                "provider profile cannot be combined with loose overrides: "
                + ", ".join(conflicts)
            )
        try:
            profile = PROVIDER_PROFILES[normalized_profile]
        except KeyError as exc:
            raise ValueError(
                f"unknown provider profile {normalized_profile!r}; "
                f"expected one of {', '.join(PROFILE_IDS)}"
            ) from exc
        active_env = environ if environ is not None else os.environ
        resolved_model = _profile_model(profile, active_env)
        resolved_vlm_model = _profile_model(
            profile,
            active_env,
            vision=True,
            fallback=resolved_model,
        )
        resolved_base_url = (
            _clean(active_env.get(profile.base_url_env, ""))
            if profile.base_url_env
            else None
        ) or _clean(profile.default_base_url)
        return ResolvedProviderSettings(
            profile_id=profile.profile_id,
            provider=profile.provider,
            model_name=resolved_model,
            vlm_provider=profile.vlm_provider,
            vlm_model=resolved_vlm_model,
            llm_wire_api=profile.llm_wire_api,
            vlm_wire_api=profile.vlm_wire_api,
            base_url=resolved_base_url,
            vlm_base_url=resolved_base_url,
        )

    resolved_provider = _clean(provider)
    if not resolved_provider:
        raise ValueError(
            "Loose model settings require an explicit provider; otherwise use a named profile"
        )
    resolved_provider = resolved_provider.lower()
    resolved_model = _clean(model_name)
    if not resolved_model:
        if resolved_provider == "gemini":
            resolved_model = "gemini-3.7-flash"
        elif resolved_provider == "qwen_local":
            active_env = environ if environ is not None else os.environ
            resolved_model = _clean(active_env.get("QWEN_LOCAL_MODEL")) or "eviscene"
        else:
            raise ValueError("An explicit model name is required for this provider")
    return ResolvedProviderSettings(
        profile_id=None,
        provider=resolved_provider.lower(),
        model_name=resolved_model,
        vlm_provider=(_clean(vlm_provider) or resolved_provider).lower(),
        vlm_model=_clean(vlm_model) or resolved_model,
        llm_wire_api=_clean(llm_wire_api),
        vlm_wire_api=_clean(vlm_wire_api),
        base_url=None,
        vlm_base_url=None,
    )
