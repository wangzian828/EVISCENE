# -*- coding: utf-8 -*-
"""CLI entry point for running image verification."""
from __future__ import annotations

import argparse
import asyncio

from eviscene.workflow import VerificationWorkflow, WorkflowConfig
from eviscene.provider_profiles import PROFILE_IDS
from eviscene.storage import default_trace_dir


def main():
    parser = argparse.ArgumentParser(prog="eviscene", description="EviScene image factuality assessment")
    parser.add_argument("image_path", help="Path to the image to verify")
    parser.add_argument(
        "--profile",
        choices=PROFILE_IDS,
        default=None,
        help="Provider profile (default: student-qwen3.5-local; cloud profiles are opt-in).",
    )
    parser.add_argument("--provider", default=None, help="Loose LLM provider")
    parser.add_argument("--model", default=None, help="Loose model name")
    parser.add_argument("--vlm-provider", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument(
        "--image-access-mode",
        choices=["direct_multimodal", "separate_vlm"],
        default=None,
        help=(
            "Whether the main LLM receives the image directly or only receives "
            "structured VLM observations."
        ),
    )
    parser.add_argument(
        "--llm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Wire protocol. Gemini accepts interactions only.",
    )
    parser.add_argument(
        "--vlm-wire-api",
        default=None,
        choices=["interactions", "responses", "chat_completions"],
        help="Vision wire protocol. Gemini accepts interactions only.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_trace_dir(),
        help="Trace output directory",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="Timeout in seconds")
    parser.add_argument("--no-trace", action="store_true", help="Don't save trace files")
    args = parser.parse_args()

    config = WorkflowConfig(
        profile_id=args.profile,
        provider=args.provider,
        model_name=args.model,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        image_access_mode=args.image_access_mode or "direct_multimodal",
        llm_wire_api=args.llm_wire_api,
        vlm_wire_api=args.vlm_wire_api,
        output_dir=args.output_dir,
        timeout=args.timeout,
        save_traces=not args.no_trace,
    )

    workflow = VerificationWorkflow(config)
    result = asyncio.run(workflow.run_single(args.image_path))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Image: {args.image_path}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Stop reason: {result.get('stop_reason', 'unknown')}")
    print(f"Actions: {result.get('action_count', result['total_tool_calls'])}")
    print(f"Assessment: {result['overall_assessment']}")
    report = result.get("fact_check_report")
    if isinstance(report, dict):
        print(f"Fact-check headline: {report.get('headline', '')}")
        print(f"Fact-check conclusion: {report.get('verdict_summary', '')}")
    print(f"Time: {result['time_taken']:.1f}s")
    print(f"Tool calls: {result['total_tool_calls']}")
    print(f"Tokens: {result['token_usage']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
