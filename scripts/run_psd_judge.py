"""Localize a training failure or judge/finalize cached PSD repair episodes."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from eviscene_training.io import load_json, sha256_file
from eviscene_training.psd_gemini_judge import (
    _atomic_json, judge_run, localize_failure, require_training_case,
)


async def run(args):
    from eviscene.integrations.gemini import GeminiInteractionsClient
    from eviscene_training.psd_repair_verifier import verify_source_rollout_failure
    trace, gold = load_json(args.trace), load_json(args.gold)
    require_training_case(trace, args.train_cases)
    from eviscene_training.psd_source_review import source_review_reference
    source_review = source_review_reference({"source_task_review": {
        "path": str(args.source_review), "sha256": sha256_file(args.source_review)
    }}) if getattr(args, "source_review", None) else None
    if not verify_source_rollout_failure(trace, gold=gold, source_task_review=source_review)["passed"]:
        raise ValueError("PSD source is not a verified failed training rollout")
    async with GeminiInteractionsClient(timeout=240, max_retries=2) as client:
        if args.mode == "localize":
            result = await localize_failure(client, trace, gold=gold, image_path=args.image,
                model=args.model, cache_dir=args.output.parent / "judge-cache")
            _atomic_json(args.output, result)
        else:
            if not args.run_dir:
                raise ValueError("--run-dir is required for finalize")
            result = await judge_run(run_dir=args.run_dir, source_trace_path=args.trace,
                gold_path=args.gold, image_path=args.image, train_cases_path=args.train_cases,
                model=args.model, client=client)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["localize", "finalize"])
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--train-cases", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "localize" and not args.output:
        parser.error("localize requires --output")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "blocked_unresolved_verifiers":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
