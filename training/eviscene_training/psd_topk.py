"""Collect frozen self-teacher top-k targets through a local vLLM server.

The collector mirrors the published PSD implementation's forced-token scoring
path: send ``teacher_prompt_ids + completion_ids`` as one token-ID prompt,
request prompt log-probabilities, and retain the distribution at each
completion position.  It never decodes and re-encodes a banked completion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .artifact_receipts import artifact_identity
from .psd import (
    PSD_TARGET_SCHEMA_VERSION,
    PSD_TOPK_CACHE_SCHEMA_VERSION,
    _completion_ids,
    _prompt_ids,
    _teacher_identity,
    validate_topk_by_position,
)
from .psd_modality import require_text_only_psd
from .psd_media import media_digest
from .psd_capture_semantics import RAW_POLICY_LOGPROBS


PSD_TOPK_COLLECTION_SCHEMA_VERSION = "ifv-psd-topk-collection-v1"
PSD_UPSTREAM_REFERENCE_COMMIT = "778be78bdac582b51a975ff819046583aad383e0"

Requester = Callable[[str, Mapping[str, Any] | None, float], Mapping[str, Any]]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json(list(token_ids)).encode("utf-8")
    ).hexdigest()


def _request_json(
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> Mapping[str, Any]:
    # The scorer is a loopback-only service.  Explicitly bypass cluster proxy
    # variables so a local request can never leave the machine.
    opener = build_opener(ProxyHandler({}))
    request = Request(
        url,
        data=(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM HTTP {exc.code} from {url}: {detail[:2000]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"vLLM request failed for {url}: {exc}") from exc
    value = json.loads(raw) if raw else None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"vLLM returned a non-object response from {url}")
    return value


def _int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty token-ID list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field}[{index}] must be an integer")
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be an integer") from exc
    return result


def _ranked_logprobs(entries: Any) -> list[tuple[int, float]]:
    """Normalize vLLM's integer-keyed prompt-logprob object to ranked pairs."""

    raw_items: list[tuple[Any, Any]]
    if isinstance(entries, Mapping):
        raw_items = list(entries.items())
    elif isinstance(entries, list):
        raw_items = [(None, item) for item in entries]
    else:
        raise ValueError("prompt-logprob position is not a mapping/list")

    ranked: list[tuple[int | None, float, int]] = []
    for raw_key, raw_value in raw_items:
        value = raw_value if isinstance(raw_value, Mapping) else {}
        token_value = value.get("token_id", raw_key)
        if isinstance(token_value, str) and token_value.startswith("token_id:"):
            token_value = token_value.split(":", 1)[1]
        if isinstance(token_value, bool):
            raise ValueError("prompt-logprob token ID is boolean")
        try:
            token_id = int(token_value)
            logprob = float(value.get("logprob", raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt-logprob entry is malformed") from exc
        if not math.isfinite(logprob):
            raise ValueError("prompt-logprob entry is non-finite")
        raw_rank = value.get("rank")
        try:
            rank = int(raw_rank) if raw_rank is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt-logprob rank is malformed") from exc
        ranked.append((rank, logprob, token_id))

    if len({item[2] for item in ranked}) != len(ranked):
        raise ValueError("teacher returned duplicate prompt-logprob token IDs")
    # vLLM includes the forced token even when it falls outside the requested
    # top-k.  Rank/log-probability sorting removes that extra token and recovers
    # the actual teacher top-k, matching upstream ``entries[:topk]``.
    ranked.sort(
        key=lambda item: (
            item[0] if item[0] is not None and item[0] > 0 else math.inf,
            -item[1],
            item[2],
        )
    )
    return [(token_id, logprob) for _, logprob, token_id in ranked]


def normalized_topk(entries: Any, *, topk: int) -> list[list[float | int]]:
    """Return the renormalized teacher distribution over exactly ``topk`` IDs."""

    if topk < 1:
        raise ValueError("topk must be positive")
    chosen = _ranked_logprobs(entries)[:topk]
    if len(chosen) != topk:
        raise ValueError(f"expected {topk} teacher candidates, got {len(chosen)}")
    maximum = max(logprob for _, logprob in chosen)
    weights = [math.exp(logprob - maximum) for _, logprob in chosen]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("teacher top-k normalization failed")
    normalized = [
        [token_id, weight / denominator]
        for (token_id, _), weight in zip(chosen, weights, strict=True)
    ]
    if not math.isclose(
        sum(float(item[1]) for item in normalized),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("normalized teacher top-k does not sum to one")
    return normalized


def _validated_targets(
    targets_path: Path,
    *,
    profile: Mapping[str, Any],
    checkpoint_manifest_sha256: str,
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    targets = load_jsonl(targets_path)
    if not targets:
        raise ValueError("PSD target input is empty")
    profile_id = _text(profile.get("profile_id"))
    context_length = profile.get("context_length")
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise ValueError("serving profile context_length is invalid")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row_index, target in enumerate(targets):
        target_id = _text(target.get("target_id"))
        if not target_id:
            raise ValueError(f"target row {row_index} is missing target_id")
        if target_id in seen:
            raise ValueError(f"duplicate PSD target_id: {target_id}")
        seen.add(target_id)
        if _text(target.get("schema_version")) != PSD_TARGET_SCHEMA_VERSION:
            raise ValueError(f"{target_id}: target schema is invalid")
        teacher = _teacher_identity(target)
        if teacher["provider"] != "qwen_local":
            raise ValueError(f"{target_id}: frozen teacher must use qwen_local")
        if teacher["model"] != profile_id:
            raise ValueError(f"{target_id}: frozen teacher model/profile mismatch")
        if teacher["checkpoint"] != checkpoint_path:
            raise ValueError(f"{target_id}: frozen teacher checkpoint mismatch")
        if teacher["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256:
            raise ValueError(
                f"{target_id}: frozen teacher checkpoint-manifest mismatch"
            )
        teacher_prompt_ids = _prompt_ids(target, "teacher_prompt_ids")
        completion_ids = _completion_ids(target)
        require_text_only_psd(target, teacher_prompt_ids, completion_ids)
        if len(teacher_prompt_ids) + len(completion_ids) + 1 > context_length:
            raise ValueError(
                f"{target_id}: forced scoring request exceeds served context"
            )
        validated.append(
            {
                "target": target,
                "target_id": target_id,
                "teacher": teacher,
                "teacher_prompt_ids": teacher_prompt_ids,
                "completion_ids": completion_ids,
            }
        )
    return validated


def _validate_attestation(
    *,
    serving_profile_path: Path,
    checkpoint_manifest_path: Path,
) -> tuple[dict[str, Any], str, str]:
    profile = load_json(serving_profile_path)
    checkpoint = load_json(checkpoint_manifest_path)
    checkpoint_sha256 = sha256_file(checkpoint_manifest_path).casefold()
    if profile.get("schema_version") != "ifv-qwen-serving-profile-v1":
        raise ValueError("policy serving profile schema is invalid")
    if _text(profile.get("engine")) != "vllm":
        raise ValueError("PSD top-k collector requires a vLLM serving profile")
    if _text(profile.get("wire_api")) != "chat_completions":
        raise ValueError("policy serving profile wire API is invalid")
    if checkpoint.get("schema_version") != "ifv-qwen-checkpoint-manifest-v1":
        raise ValueError("round-start checkpoint manifest schema is invalid")
    checkpoint_path = _text(
        (checkpoint.get("checkpoint") or {}).get("path")
        if isinstance(checkpoint.get("checkpoint"), Mapping)
        else ""
    )
    if not checkpoint_path:
        raise ValueError("round-start checkpoint path is missing")
    if _text(profile.get("model_path")) != checkpoint_path:
        raise ValueError("served model path does not match round-start checkpoint")
    if _text(profile.get("checkpoint_manifest_sha256")).casefold() != checkpoint_sha256:
        raise ValueError("serving profile is not bound to checkpoint manifest")
    base_url = _text(profile.get("base_url")).rstrip("/")
    if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("PSD top-k collector only accepts a loopback vLLM URL")
    return profile, checkpoint_path, checkpoint_sha256


def _cache_record(
    *,
    scored: Mapping[str, Any],
    profile: Mapping[str, Any],
    topk: int,
    response: Mapping[str, Any],
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vLLM must return exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("vLLM completion choice is malformed")
    combined_ids = [
        *scored["teacher_prompt_ids"],
        *scored["completion_ids"],
    ]
    returned_prompt_ids = _int_list(
        choice.get("prompt_token_ids"),
        field="response.prompt_token_ids",
    )
    if returned_prompt_ids != combined_ids:
        raise ValueError(
            "vLLM response.prompt_token_ids differ from the forced prompt"
        )
    prompt_logprobs = choice.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list):
        raise ValueError("vLLM response omitted prompt_logprobs")
    if len(prompt_logprobs) != len(combined_ids):
        raise ValueError("vLLM prompt_logprobs length does not match forced prompt")

    distributions: list[list[list[float | int]]] = []
    start = len(scored["teacher_prompt_ids"])
    for offset in range(len(scored["completion_ids"])):
        position = start + offset
        entries = prompt_logprobs[position]
        if entries is None:
            raise ValueError(
                f"vLLM omitted teacher distribution at position {position}"
            )
        distributions.append(normalized_topk(entries, topk=topk))
    validate_topk_by_position(
        scored["completion_ids"],
        distributions,
        topk=topk,
    )
    teacher = scored["teacher"]
    return {
        "schema_version": PSD_TOPK_CACHE_SCHEMA_VERSION,
        "target_id": scored["target_id"],
        "teacher_provider": teacher["provider"],
        "teacher_model": teacher["model"],
        "teacher_checkpoint": teacher["checkpoint"],
        "teacher_checkpoint_manifest_sha256": teacher[
            "checkpoint_manifest_sha256"
        ],
        "teacher_prompt_sha256": _token_ids_sha256(
            scored["teacher_prompt_ids"]
        ),
        "completion_sha256": _token_ids_sha256(scored["completion_ids"]),
        **({"media_sha256": media_digest(scored["target"]["psd_media"])}
           if scored["target"].get("psd_media") else {}),
        "teacher_topk_by_position": distributions,
        "teacher_logprob_semantics": RAW_POLICY_LOGPROBS,
        "collection": {
            "engine": "vllm",
            "profile_id": _text(profile.get("profile_id")),
            "method": "forced_token_ids_prompt_logprobs",
            "topk": topk,
        },
    }


def _validate_existing_cache(
    *,
    cache_path: Path,
    targets_by_id: Mapping[str, Mapping[str, Any]],
    topk: int,
) -> dict[str, Mapping[str, Any]]:
    existing: dict[str, Mapping[str, Any]] = {}
    for row_index, row in enumerate(load_jsonl(cache_path)):
        target_id = _text(row.get("target_id"))
        if not target_id or target_id not in targets_by_id:
            raise ValueError(f"existing cache row {row_index} has unknown target_id")
        if target_id in existing:
            raise ValueError(f"existing cache has duplicate target_id: {target_id}")
        scored = targets_by_id[target_id]
        teacher = scored["teacher"]
        expected = {
            "schema_version": PSD_TOPK_CACHE_SCHEMA_VERSION,
            "teacher_logprob_semantics": RAW_POLICY_LOGPROBS,
            "teacher_provider": teacher["provider"],
            "teacher_model": teacher["model"],
            "teacher_checkpoint": teacher["checkpoint"],
            "teacher_checkpoint_manifest_sha256": teacher[
                "checkpoint_manifest_sha256"
            ],
            "teacher_prompt_sha256": _token_ids_sha256(
                scored["teacher_prompt_ids"]
            ),
            "completion_sha256": _token_ids_sha256(scored["completion_ids"]),
        }
        for field, expected_value in expected.items():
            if _text(row.get(field)) != expected_value:
                raise ValueError(
                    f"existing cache {target_id} has mismatched {field}"
                )
        if scored["target"].get("psd_media") and row.get("media_sha256") != media_digest(scored["target"]["psd_media"]):
            raise ValueError("existing cache media mismatch")
        validate_topk_by_position(
            scored["completion_ids"],
            row.get("teacher_topk_by_position"),
            topk=topk,
        )
        existing[target_id] = row
    return existing


def _append_cache_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def collect_psd_topk_cache(
    *,
    targets_path: Path,
    serving_profile_path: Path,
    checkpoint_manifest_path: Path,
    output_dir: Path,
    topk: int = 20,
    retries: int = 3,
    timeout: float = 1800.0,
    limit: int | None = None,
    requester: Requester | None = None,
    retry_sleep: Callable[[float], None] = time.sleep,
    backend: str = "vllm",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Collect or resume an attested, per-target frozen-teacher top-k cache."""

    if topk < 1:
        raise ValueError("topk must be positive")
    if retries < 1:
        raise ValueError("retries must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    profile, checkpoint_path, checkpoint_sha256 = _validate_attestation(
        serving_profile_path=serving_profile_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
    )
    scored_targets = _validated_targets(
        targets_path,
        profile=profile,
        checkpoint_manifest_sha256=checkpoint_sha256,
        checkpoint_path=checkpoint_path,
    )
    targets_by_id = {row["target_id"]: row for row in scored_targets}
    if backend not in {"vllm", "transformers"}:
        raise ValueError("unsupported PSD teacher backend")
    if backend == "vllm" and any(row["target"].get("psd_media") for row in scored_targets):
        raise ValueError("multimodal exact-token scoring requires --backend transformers")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "teacher_topk_cache.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        prior = load_json(manifest_path)
        prior_inputs = prior.get("inputs")
        prior_inputs = prior_inputs if isinstance(prior_inputs, Mapping) else {}
        if prior_inputs.get("targets_identity") is not None:
            if prior_inputs["targets_identity"] != artifact_identity(targets_path):
                raise ValueError("existing collection manifest has mismatched targets_identity")
        else:
            expected_bindings = {
                "targets_sha256": sha256_file(targets_path),
                "serving_profile_sha256": sha256_file(serving_profile_path),
                "checkpoint_manifest_sha256": checkpoint_sha256,
            }
            for field, expected_value in expected_bindings.items():
                if _text(prior_inputs.get(field)).casefold() != expected_value.casefold():
                    raise ValueError(
                        f"existing collection manifest has mismatched {field}"
                    )
    existing = _validate_existing_cache(
        cache_path=cache_path,
        targets_by_id=targets_by_id,
        topk=topk,
    )

    base_url = _text(profile.get("base_url")).rstrip("/")
    pending = [row for row in scored_targets if row["target_id"] not in existing]
    selected = pending[:limit] if limit is not None else pending
    request = requester or _request_json
    teacher = None
    if selected and backend == "transformers":
        from .psd_hf_teacher import FrozenTeacher
        teacher = FrozenTeacher(profile, device=device)
    if selected and backend == "vllm":
        models = request(f"{base_url}/models", None, timeout)
        cards = models.get("data")
        if not isinstance(cards, list) or not any(
            isinstance(card, Mapping)
            and _text(card.get("id")) == _text(profile.get("profile_id"))
            for card in cards
        ):
            raise RuntimeError("served profile ID is absent from vLLM /models")
    failures: list[dict[str, Any]] = []
    collected = 0
    for scored in selected:
        error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                combined_ids = [
                    *scored["teacher_prompt_ids"],
                    *scored["completion_ids"],
                ]
                response = teacher.score(scored, topk) if teacher else request(
                    f"{base_url}/completions",
                    {
                        "model": _text(profile.get("profile_id")),
                        "prompt": combined_ids,
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "prompt_logprobs": topk,
                        "return_token_ids": True,
                        "return_tokens_as_token_ids": True,
                    },
                    timeout,
                )
                row = _cache_record(
                    scored=scored,
                    profile=profile,
                    topk=topk,
                    response=response,
                )
                row["collection"]["engine"] = backend
                _append_cache_row(cache_path, row)
                existing[scored["target_id"]] = row
                collected += 1
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - persisted for bounded retry
                error = exc
                if attempt < retries:
                    retry_sleep(float(min(2**attempt, 8)))
        if error is not None:
            failures.append(
                {
                    "target_id": scored["target_id"],
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "attempts": retries,
                }
            )

    write_jsonl(output_dir / "failures.jsonl", failures)
    remaining = len(scored_targets) - len(existing)
    status = (
        "ready_for_materialization"
        if remaining == 0
        else "blocked_collection_errors"
        if failures
        else "incomplete_limit"
    )
    manifest = {
        "schema_version": PSD_TOPK_COLLECTION_SCHEMA_VERSION,
        "status": status,
        "topk": topk,
        "method": "forced_token_ids_prompt_logprobs",
        "backend": backend,
        "upstream_reference": {
            "repository": "essamsleiman/psd",
            "commit": PSD_UPSTREAM_REFERENCE_COMMIT,
        },
        "inputs": {
            "targets": str(targets_path.resolve()),
            "targets_identity": artifact_identity(targets_path),
            "serving_profile": str(serving_profile_path.resolve()),
            "serving_profile_sha256": sha256_file(serving_profile_path),
            "checkpoint_manifest": str(checkpoint_manifest_path.resolve()),
            "checkpoint_manifest_sha256": checkpoint_sha256,
        },
        "teacher": {
            "provider": "qwen_local",
            "model": _text(profile.get("profile_id")),
            "checkpoint": checkpoint_path,
            "base_url": base_url,
        },
        "counts": {
            "targets": len(scored_targets),
            "cached_before_run": len(existing) - collected,
            "collected_this_run": collected,
            "cached_total": len(existing),
            "failures_this_run": len(failures),
            "remaining": remaining,
        },
        "artifacts": {
            "cache": "teacher_topk_cache.jsonl",
            "failures": "failures.jsonl",
        },
    }
    write_json(manifest_path, manifest)
    return manifest
