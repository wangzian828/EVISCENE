"""Select a deterministic held-out canary before observing model outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from eviscene_training.io import load_jsonl, require_new_or_empty, sha256_file, write_json


def prepare(*, benchmark: Path, output_dir: Path, count: int, salt: str) -> dict:
    benchmark = benchmark.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rows = load_jsonl(benchmark)
    case_ids = [str(row.get("case_id") or "").strip() for row in rows]
    if not rows or any(not case_id for case_id in case_ids):
        raise ValueError("held-out benchmark must contain nonempty case IDs")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("held-out benchmark case IDs must be unique")
    if type(count) is not int or not 1 <= count <= len(case_ids):
        raise ValueError("paired evaluation count is outside the benchmark")
    salt = str(salt or "").strip()
    if not salt:
        raise ValueError("paired evaluation selection salt must be nonempty")

    selected = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(
            f"{salt}:{case_id}".encode("utf-8")
        ).hexdigest(),
    )[:count]
    require_new_or_empty(output_dir)
    case_list = output_dir / "case-list.txt"
    case_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    result = {
        "schema_version": "ifv-psd-paired-evaluation-selection-v1",
        "selection_before_outcomes": True,
        "selection_method": "ascending_sha256(salt + ':' + case_id)",
        "salt": salt,
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_cases": len(case_ids),
        "selected_cases": len(selected),
        "case_list": str(case_list),
        "case_list_sha256": sha256_file(case_list),
        "case_ids": selected,
    }
    write_json(output_dir / "selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--salt", default="psd-round1-heldout-v1")
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
