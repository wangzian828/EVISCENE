"""Select complete preservation episodes for a smaller PSD training bank."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _key(row: dict[str, Any]) -> tuple[str, str]:
    case_id, episode_id = row.get("case_id"), row.get("episode_id")
    if not isinstance(case_id, str) or not case_id or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("preservation rows require case_id and episode_id")
    return case_id, episode_id


def _expected(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    episodes: dict[tuple[str, str], tuple[str, ...]] = {}
    cases: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = _key(row)
            if key in episodes or key[0] in cases:
                raise ValueError("preservation source must have one episode per case")
            if row.get("verified_full_task") is not True or row.get("strict_trace_audit_pass") is not True:
                raise ValueError("preservation source episode is not strictly verified")
            steps = row.get("preservation_steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("preservation source episode has no steps")
            ids = tuple(step.get("step_id") for step in steps)
            if any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
                raise ValueError("preservation source step IDs missing or duplicated")
            if sum(":judgment:" in s for s in ids) != 1:
                raise ValueError("preservation source requires one judgment step")
            episodes[key] = ids
            cases.add(key[0])
    if not episodes:
        raise ValueError("no preservation episodes")
    return episodes


def _select(episodes: dict[tuple[str, str], tuple[str, ...]], cap: int, seed: str):
    lengths = sorted(map(len, episodes.values()))
    bounds = tuple(lengths[len(lengths) * q // 4] for q in (1, 2, 3))
    groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for key, ids in episodes.items():
        groups[sum(len(ids) > bound for bound in bounds)].append(key)
    total = sum(lengths)
    selected: set[tuple[str, str]] = set()
    strata = {}
    for bucket, keys in sorted(groups.items()):
        source_steps = sum(len(episodes[key]) for key in keys)
        quota = cap * source_steps / total
        order = sorted(keys, key=lambda key: (
            hashlib.sha256(f"{seed}\0{key[0]}\0{key[1]}".encode()).digest(), key))
        chosen = []
        mass = 0
        for key in order:
            if mass >= quota:
                break
            step_count = len(episodes[key])
            if abs(mass + step_count - quota) > abs(mass - quota):
                break
            chosen.append(key)
            mass += step_count
        selected.update(chosen)
        strata[str(bucket)] = {"source_episodes": len(keys), "source_steps": source_steps,
                               "target_steps": round(quota, 2),
                               "selected_episodes": len(chosen), "selected_steps": mass}
    # A small bank may have a stratum quota shorter than one episode. Repair
    # the global count by toggling whole episodes; never split a trajectory.
    tolerance = max(max(lengths), math.ceil(cap * .05))
    while True:
        mass = sum(len(episodes[key]) for key in selected)
        if abs(mass - cap) <= tolerance:
            break
        choices = []
        for key in episodes:
            new_mass = mass + (len(episodes[key]) if key not in selected else -len(episodes[key]))
            if abs(new_mass - cap) >= abs(mass - cap):
                continue
            bucket = sum(len(episodes[key]) > bound for bound in bounds)
            new_bucket_mass = strata[str(bucket)]["selected_steps"] + (
                len(episodes[key]) if key not in selected else -len(episodes[key]))
            bucket_deviation = abs(new_bucket_mass - strata[str(bucket)]["target_steps"])
            choices.append((abs(new_mass - cap), bucket_deviation,
                            hashlib.sha256(f"{seed}\0{key[0]}\0{key[1]}".encode()).digest(), key,
                            bucket, new_bucket_mass))
        if not choices:
            break
        _, _, _, key, bucket, new_bucket_mass = min(choices)
        if key in selected:
            selected.remove(key)
            strata[str(bucket)]["selected_episodes"] -= 1
        else:
            selected.add(key)
            strata[str(bucket)]["selected_episodes"] += 1
        strata[str(bucket)]["selected_steps"] = new_bucket_mass
    return selected, {"length_quartile_boundaries": bounds, "length_strata": strata}


def balance_psd_targets(
    *, source: Path, preservation_episodes_source: Path, output_dir: Path,
    preserve_target_cap: int | None = None, seed: str = "psd-whole-episode-v2",
) -> dict[str, Any]:
    """Keep all repairs and seeded, length-stratified *whole* successful episodes.

    The assembled preservation file is the completeness authority. Both files
    are read once; selected original target lines are then copied by offset.
    No gold labels, teacher scores, provider calls, or payload hashes are used.
    """
    if not seed:
        raise ValueError("selection seed must be nonempty")
    source, preservation_episodes_source, output_dir = (
        source.resolve(), preservation_episodes_source.resolve(), output_dir.resolve())
    if not source.is_file() or not preservation_episodes_source.is_file():
        raise FileNotFoundError("target or assembled preservation source missing")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    expected = _expected(preservation_episodes_source)
    offsets: dict[tuple[str, str], list[int]] = defaultdict(list)
    step_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    repairs: list[int] = []
    seen: set[str] = set()
    with source.open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            target_id, kind = row.get("target_id"), row.get("kind")
            if not isinstance(target_id, str) or not target_id or target_id in seen:
                raise ValueError("PSD target IDs must be present and unique")
            seen.add(target_id)
            if kind == "repair":
                repairs.append(position)
            elif kind == "preserve":
                key = _key(row)
                verification = row.get("verification") or {}
                if not all(verification.get(field) is True for field in
                           ("local_pass", "full_episode_pass", "strict_trace_audit_pass")):
                    raise ValueError("preservation target lacks strict verification")
                step_id = (row.get("repair_site") or {}).get("step_id")
                if not isinstance(step_id, str) or not step_id:
                    raise ValueError("preservation target step ID missing")
                offsets[key].append(position)
                step_ids[key].append(step_id)
            else:
                raise ValueError(f"unknown PSD target kind: {kind!r}")
    if set(expected) != set(step_ids):
        raise ValueError("assembled and target preservation episodes differ")
    for key, ids in expected.items():
        if tuple(step_ids[key]) != ids:
            raise ValueError(f"preservation episode incomplete or out of order: {key}")
    repair_count = len(repairs)
    available = sum(map(len, expected.values()))
    if not repair_count:
        raise ValueError("balanced PSD training requires repair targets")
    cap = repair_count if preserve_target_cap is None else preserve_target_cap
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= available:
        raise ValueError("preservation cap must be positive and available")
    selected, detail = _select(expected, cap, seed)
    preserve_count = sum(len(expected[key]) for key in selected)
    tolerance = max(max(map(len, expected.values())), math.ceil(cap * .05))
    if not selected or abs(preserve_count - cap) > tolerance:
        raise ValueError("whole-episode selection outside declared count tolerance")
    chosen_offsets = sorted(repairs + [position for key in selected for position in offsets[key]])
    if len(chosen_offsets) != repair_count + preserve_count:
        raise RuntimeError("whole-episode target selection is not unique")
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "targets.jsonl"
    with source.open("rb") as inp, destination.open("xb") as out:
        for position in chosen_offsets:
            inp.seek(position)
            out.write(inp.readline())
    manifest = {
        "schema_version": "ifv-psd-whole-episode-target-selection-v2",
        "selection_policy": "all_repairs_plus_seeded_length_stratified_complete_preservation_episodes",
        "seed": seed, "source_identity": _identity(source),
        "preservation_episodes_identity": _identity(preservation_episodes_source),
        "output_identity": _identity(destination),
        "counts": {"repair_targets": repair_count, "source_preservation_targets": available,
                   "source_preservation_cases": len(expected),
                   "selected_preservation_cases": len(selected),
                   "selected_preservation_targets": preserve_count,
                   "selected_targets": repair_count + preserve_count,
                   "excluded_preservation_targets": available - preserve_count,
                   "requested_preservation_step_cap": cap,
                   "whole_episode_tolerance_steps": tolerance},
        **detail,
        "selected_episode_ids": [{"case_id": c, "episode_id": e, "steps": len(expected[(c, e)])}
                                 for c, e in sorted(selected)],
        "per_target_weights_unchanged": True, "source_targets_unchanged": True,
        "provider_calls": 0, "large_payload_hashing": False,
        "status": "prepared_for_audit_not_authorized_to_train",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
