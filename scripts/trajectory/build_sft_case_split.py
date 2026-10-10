#!/usr/bin/env python3
"""Freeze a leakage-safe train/validation split for reviewed SFT cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA_VERSION = "ifv-sft-case-split-v1"


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _index(rows: Sequence[Mapping[str, Any]], *, name: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{name} row lacks case_id")
        if case_id in result:
            raise ValueError(f"duplicate {name} case_id: {case_id}")
        result[case_id] = row
    return result


def _label(row: Mapping[str, Any]) -> str:
    label = str(row.get("label") or row.get("factual_status") or "").strip().lower()
    if label in {"supported", "refuted"}:
        return label
    verdict = str(
        row.get("expected_verdict") or row.get("gold_verdict") or ""
    ).strip().lower()
    mapping = {"real": "supported", "fake": "refuted"}
    if verdict in mapping:
        return mapping[verdict]
    raise ValueError("gold labels must be supported/refuted or real/fake")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _group_keys(
    runtime: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> set[str]:
    keys = {f"image:{str(runtime.get('image_sha256') or '').casefold()}"}
    claim_atom = gold.get("claim_atom")
    if isinstance(claim_atom, Mapping):
        subject = str(claim_atom.get("subject") or "").strip().casefold()
        event = str(claim_atom.get("event") or "").strip().casefold()
        if subject and event:
            keys.add(f"subject-event:{subject}|{event}")
    evidence = gold.get("certifying_evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, Mapping):
                continue
            for field in ("source_id", "snapshot_sha256"):
                value = str(row.get(field) or "").strip().casefold()
                if value:
                    keys.add(f"{field}:{value}")
    return keys


def build_split(
    runtime_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    validation_count: int,
    validation_supported: int,
    seed: str,
    prohibited_case_ids: set[str] | None = None,
    prohibited_image_hashes: set[str] | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    runtime = _index(runtime_rows, name="runtime")
    gold = _index(gold_rows, name="gold")
    if set(runtime) != set(gold):
        raise ValueError("runtime and gold case IDs must match exactly")
    if validation_count < 1 or validation_count >= len(runtime):
        raise ValueError("validation_count must leave non-empty train and validation")
    if validation_supported < 0 or validation_supported > validation_count:
        raise ValueError("validation_supported is outside validation_count")

    blocked_ids = prohibited_case_ids or set()
    blocked_hashes = {value.casefold() for value in (prohibited_image_hashes or set())}
    overlap_ids = sorted(set(runtime) & blocked_ids)
    overlap_hashes = sorted(
        {
            str(row.get("image_sha256") or "").casefold()
            for row in runtime.values()
        }
        & blocked_hashes
    )
    if overlap_ids or overlap_hashes:
        raise ValueError(
            "training candidates overlap prohibited evaluation data: "
            f"case_ids={overlap_ids[:5]}, image_hashes={overlap_hashes[:5]}"
        )

    labels = {case_id: _label(gold[case_id]) for case_id in gold}

    union = _UnionFind(runtime)
    members_by_key: Dict[str, list[str]] = defaultdict(list)
    for case_id in runtime:
        for key in _group_keys(runtime[case_id], gold[case_id]):
            members_by_key[key].append(case_id)
    for members in members_by_key.values():
        for member in members[1:]:
            union.union(members[0], member)

    components: Dict[str, list[str]] = defaultdict(list)
    for case_id in runtime:
        components[union.find(case_id)].append(case_id)
    groups: list[Dict[str, Any]] = []
    for members in components.values():
        ordered = sorted(members)
        group_id = hashlib.sha256(
            json.dumps(ordered, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        groups.append(
            {
                "group_id": group_id,
                "members": ordered,
                "count": len(ordered),
                "supported": sum(labels[item] == "supported" for item in ordered),
            }
        )
    groups.sort(
        key=lambda group: hashlib.sha256(
            f"{seed}:{group['group_id']}".encode("utf-8")
        ).hexdigest()
    )

    target = (validation_count, validation_supported)
    choices: Dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for index, group in enumerate(groups):
        next_choices = dict(choices)
        for (count, supported), selected in choices.items():
            state = (
                count + int(group["count"]),
                supported + int(group["supported"]),
            )
            if state[0] > target[0] or state[1] > target[1]:
                continue
            next_choices.setdefault(state, selected + (index,))
        choices = next_choices
    if target not in choices:
        raise ValueError(
            "group constraints cannot satisfy requested validation count/labels"
        )
    validation_groups = {groups[index]["group_id"] for index in choices[target]}
    group_by_case = {
        case_id: str(group["group_id"])
        for group in groups
        for case_id in group["members"]
    }
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "image_sha256": str(runtime[case_id].get("image_sha256") or ""),
            "split_group_id": group_by_case[case_id],
            "split": (
                "validation"
                if group_by_case[case_id] in validation_groups
                else "train"
            ),
        }
        for case_id in sorted(runtime)
    ]
    split_labels = {
        split: Counter(labels[row["case_id"]] for row in rows if row["split"] == split)
        for split in ("train", "validation")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "case_count": len(rows),
        "group_count": len(groups),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_label_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in split_labels.items()
        },
        "validation_group_ids": sorted(validation_groups),
    }
    return rows, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--prohibited-runtime", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--validation-supported", type=int, default=2)
    parser.add_argument("--seed", default="ifv-qwen35-sft-v1")
    return parser.parse_args()


def main() -> int:
    args = _args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows = _load_jsonl(args.runtime)
    gold_rows = _load_jsonl(args.gold)
    prohibited_rows = (
        _load_jsonl(args.prohibited_runtime)
        if args.prohibited_runtime is not None
        else []
    )
    rows, summary = build_split(
        runtime_rows,
        gold_rows,
        validation_count=args.validation_count,
        validation_supported=args.validation_supported,
        seed=args.seed,
        prohibited_case_ids={str(row.get("case_id") or "") for row in prohibited_rows},
        prohibited_image_hashes={
            str(row.get("image_sha256") or "") for row in prohibited_rows
        },
    )
    split_path = output_dir / "case_split.jsonl"
    split_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **summary,
        "inputs": {
            "runtime": {"path": str(args.runtime.resolve()), "sha256": _sha256(args.runtime)},
            "gold": {"path": str(args.gold.resolve()), "sha256": _sha256(args.gold)},
            "prohibited_runtime": (
                {
                    "path": str(args.prohibited_runtime.resolve()),
                    "sha256": _sha256(args.prohibited_runtime),
                }
                if args.prohibited_runtime is not None
                else None
            ),
        },
        "artifacts": {
            "case_split": "case_split.jsonl",
            "case_split_sha256": _sha256(split_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
