from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def rollout_specs(
    samples: list[Dict[str, Any]],
    runtime_cases: list[Any],
    *,
    rollouts_per_case: int,
    base_sampling_seed: int,
    policy_revision: str,
    model: str,
    episode_namespace: str | None = None,
) -> list[Dict[str, Any]]:
    specs: list[Dict[str, Any]] = []
    for sample, runtime_case in zip(samples, runtime_cases):
        case_id = str(runtime_case.case_id)
        group_material = {
            "case_id": case_id,
            "policy_revision": policy_revision,
            "model": model,
            "base_sampling_seed": int(base_sampling_seed),
            "group_size": rollouts_per_case,
        }
        prompt_group_id = "pg-" + hashlib.sha256(
            json.dumps(
                group_material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        for rollout_index in range(rollouts_per_case):
            sampling_seed = int.from_bytes(
                hashlib.sha256(
                    f"{base_sampling_seed}:{case_id}:{rollout_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            ) & 0x7FFFFFFF
            episode_id = case_id
            if rollouts_per_case > 1 or episode_namespace:
                episode_id = (
                    f"{case_id[:140]}--{prompt_group_id[3:11]}"
                    f"--r{rollout_index:03d}"
                )
                if episode_namespace:
                    episode_id = (
                        f"{episode_namespace}-{episode_id}"
                    )
            specs.append(
                {
                    "sample": sample,
                    "runtime_case": runtime_case,
                    "case_id": case_id,
                    "episode_id": episode_id,
                    "prompt_group_id": prompt_group_id,
                    "rollout_index": rollout_index,
                    "sampling_seed": sampling_seed,
                    "group_size": rollouts_per_case,
                }
            )
    return specs
