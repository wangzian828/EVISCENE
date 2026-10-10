"""Run the provider-neutral image-only direct-QA baseline.

The model receives only:

    one image + the shared direct-QA prompt

Dataset claims, labels, source URLs, capability cells, and generation prompts
are never included in the model request.  Gold labels are used only for local
evaluation after the response has been received.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from eviscene.integrations.gemini import GeminiInteractionsClient, extract_text


DEFAULT_PROMPT = """You are an Image Factual Verifier.

Given only one image, identify the main real-world fact or situation presented by the image and judge whether it is correct.

Consider the image as a whole. You may use visible people, objects, actions, text, numbers, places, times, and relationships as clues, but include only details that the image actually presents. Do not force the image into a more specific event, background, identity, or relationship than it conveys.

The verdict must be based on the fact or situation identified from the image. Visual style, image quality, or apparent AI/editing artifacts are not by themselves proof that the expressed fact is true or false, and must not replace evaluation of the factual content. If the expressed fact itself concerns image generation or editing, evaluate that fact directly.

Output only JSON:
{
  "core_fact": "...",
  "verdict": "real" or "fake",
  "reason": "..."
}"""


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "model output is not a JSON object"
    return value, None


def _thought_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        chunks: list[str] = []
        for item in value.values():
            chunks.extend(_thought_text(item))
        return chunks
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        chunks = []
        for item in value:
            chunks.extend(_thought_text(item))
        return chunks
    return []


def extract_native_thought(payload: Mapping[str, Any]) -> str:
    """Extract provider-returned thought summaries, never model JSON fields."""

    chunks: list[str] = []
    for step in payload.get("steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        if str(step.get("type", "")).strip().lower() != "thought":
            continue
        for field_name in ("text", "summary", "content"):
            chunks.extend(_thought_text(step.get(field_name)))
    return "\n".join(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))


def _resolve_image_path(row: Mapping[str, Any], image_root: Path) -> Path:
    raw = str(
        row.get("unified_image_path")
        or row.get("image_path")
        or row.get("local_image_path")
        or ""
    ).strip()
    if not raw:
        raise ValueError("manifest row has no image path")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else image_root / path


def _case_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("unified_case_id")
        or row.get("case_id")
        or row.get("record_id")
        or row.get("unified_image_path")
        or row.get("image_path")
        or ""
    ).strip()


def _gold_verdict(row: Mapping[str, Any]) -> str | None:
    return {
        "supported": "real",
        "refuted": "fake",
        "real": "real",
        "fake": "fake",
    }.get(str(row.get("factual_status") or row.get("label") or "").strip().lower())


def _image_input(path: Path) -> dict[str, str]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {
        "type": "image",
        "mime_type": mime_type,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _usage_thought_tokens(payload: Mapping[str, Any]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    return int(
        usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0
    )


def _load_rows(manifest: Path, image_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {manifest}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row is not an object: {manifest}:{line_number}")
            case_id = _case_id(row)
            if not case_id:
                raise ValueError(f"manifest row has no stable case ID: {manifest}:{line_number}")
            if case_id in seen:
                raise ValueError(f"duplicate case ID {case_id!r}: {manifest}:{line_number}")
            seen.add(case_id)
            row["_resolved_image_path"] = str(_resolve_image_path(row, image_root))
            rows.append(row)
    return rows


def _stratified_sample(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    field: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample rows round-robin across one manifest category field."""

    if limit <= 0 or limit >= len(rows):
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(field) or "<empty>").strip() or "<empty>"
        groups.setdefault(value, []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit:
        added = False
        for key in keys:
            group = groups[key]
            if not group:
                continue
            selected.append(group.pop())
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    rng.shuffle(selected)
    return selected


def _load_completed_ids(results_path: Path) -> set[str]:
    if not results_path.is_file():
        return set()
    latest_status: dict[str, str] = {}
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(row.get("case_id") or "").strip()
            if case_id:
                latest_status[case_id] = str(row.get("status") or "").strip()
    return {
        case_id
        for case_id, status in latest_status.items()
        if status == "completed"
    }


async def _run_one(
    row: Mapping[str, Any],
    *,
    client: GeminiInteractionsClient,
    prompt: str,
    model: str,
    response_format: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = _case_id(row)
    relative_image = str(
        row.get("unified_image_path")
        or row.get("image_path")
        or row.get("local_image_path")
        or ""
    )
    image_path = Path(str(row["_resolved_image_path"]))
    started = time.monotonic()
    result: dict[str, Any] = {
        "case_id": case_id,
        "image_path": relative_image,
        "model": model,
        "status": "error",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gold_verdict": _gold_verdict(row),
    }
    try:
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))
        # Deliberately construct the request from only image + shared prompt.
        request_input = [_image_input(image_path), {"type": "text", "text": prompt}]
        async with semaphore:
            payload = await client.create(
                model=model,
                input=request_input,
                response_format=response_format,
                generation_config=generation_config,
                store=True,
            )
        output_text = extract_text(payload)
        parsed, parse_error = _parse_json_object(output_text)
        native_thought = extract_native_thought(payload)
        predicted = (
            str(parsed.get("verdict", "")).strip().lower()
            if parsed is not None
            else ""
        )
        gold = result["gold_verdict"]
        result.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "interaction_id": payload.get("id"),
                "interaction_status": payload.get("status"),
                "native_thought": native_thought,
                "native_thought_chars": len(native_thought),
                "native_thought_tokens": _usage_thought_tokens(payload),
                "model_output_text": output_text,
                "model_output": parsed,
                "json_parse_error": parse_error,
                "predicted_verdict": predicted or None,
                "correct": bool(gold and predicted and predicted == gold),
                "usage": payload.get("usage"),
            }
        )
    except Exception as exc:
        result.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    manifest: Path,
    output_dir: Path,
    selection_counts: Mapping[str, int],
) -> dict[str, Any]:
    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "")
        if case_id:
            latest[case_id] = record
    values = list(latest.values())
    completed = [row for row in values if row.get("status") == "completed"]
    comparable = [row for row in completed if row.get("gold_verdict") in {"real", "fake"}]
    correct = [row for row in comparable if row.get("correct")]
    thought = [row for row in completed if row.get("native_thought_chars", 0) > 0]
    valid_json = [
        row
        for row in completed
        if isinstance(row.get("model_output"), dict)
        and not row.get("json_parse_error")
    ]
    errors = [row for row in values if row.get("status") != "completed"]
    error_types = Counter(str(row.get("error_type") or "unknown") for row in errors)
    return {
        "schema_version": "ifv-direct-qa-baseline-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "thinking_level": args.thinking_level,
        "thinking_summaries": "auto",
        "max_output_tokens": args.max_output_tokens,
        "concurrency": args.concurrency,
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "input_contract": "image_only",
        "prompt_contract": "shared_direct_qa_prompt",
        "selection": {
            "stratify_by": args.stratify_by,
            "seed": args.seed,
            "category_counts": dict(selection_counts),
        },
        "counts": {
            "manifest_rows_selected": len(values),
            "completed": len(completed),
            "errors": len(errors),
            "valid_json": len(valid_json),
            "native_thought_present": len(thought),
            "native_thought_absent": len(completed) - len(thought),
            "gold_comparable": len(comparable),
            "correct": len(correct),
        },
        "accuracy_on_completed_comparable": (
            round(len(correct) / len(comparable), 6) if comparable else None
        ),
        "thought_presence_rate_on_completed": (
            round(len(thought) / len(completed), 6) if completed else None
        ),
        "error_types": dict(error_types),
        "result_file": str(output_dir / "results.jsonl"),
        "prompt_file": str(output_dir / "prompt.txt"),
        "config_file": str(output_dir / "run-config.json"),
    }


async def _run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).expanduser().resolve()
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else manifest.parent
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    prompt = (
        Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
        if args.prompt_file
        else DEFAULT_PROMPT
    ).strip()
    if not prompt:
        raise ValueError("direct-QA prompt must not be empty")

    rows = _load_rows(manifest, image_root)
    if args.case_id:
        wanted = set(args.case_id)
        rows = [row for row in rows if _case_id(row) in wanted]
    if args.stratify_by:
        rows = _stratified_sample(
            rows,
            limit=args.limit if args.limit > 0 else len(rows),
            field=args.stratify_by,
            seed=args.seed,
        )
    elif args.limit > 0:
        rows = rows[: args.limit]
    if args.resume:
        completed_ids = _load_completed_ids(results_path)
        rows = [row for row in rows if _case_id(row) not in completed_ids]
    selection_counts = Counter(
        str(row.get(args.stratify_by) or "<empty>")
        for row in rows
    ) if args.stratify_by else Counter()

    schema = {
        "type": "object",
        "properties": {
            "core_fact": {"type": "string"},
            "verdict": {"type": "string", "enum": ["real", "fake"]},
            "reason": {"type": "string"},
        },
        "required": ["core_fact", "verdict", "reason"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }
    generation_config = {
        "max_output_tokens": args.max_output_tokens,
        "thinking_level": args.thinking_level,
        "thinking_summaries": "auto",
    }
    config = {
        "schema_version": "ifv-direct-qa-baseline-config-v1",
        "model": args.model,
        "thinking_level": args.thinking_level,
        "thinking_summaries": "auto",
        "max_output_tokens": args.max_output_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
        "stratify_by": args.stratify_by,
        "seed": args.seed,
        "manifest": str(manifest),
        "image_root": str(image_root),
        "selected_rows": len(rows),
        "selection_category_counts": dict(selection_counts),
        "input_contract": "image_only",
        "request_fields": ["image", "shared_prompt"],
    }
    (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    _write_json(output_dir / "run-config.json", config)

    existing_records: list[dict[str, Any]] = []
    if results_path.is_file():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        existing_records.append(value)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    async with GeminiInteractionsClient(
        timeout=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        for row in rows:
            tasks.append(
                asyncio.create_task(
                    _run_one(
                        row,
                        client=client,
                        prompt=prompt,
                        model=args.model,
                        response_format=response_format,
                        generation_config=generation_config,
                        semaphore=semaphore,
                    )
                )
            )
        with results_path.open("a", encoding="utf-8") as handle:
            for task in asyncio.as_completed(tasks):
                result = await task
                existing_records.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "case_id": result.get("case_id"),
                            "status": result.get("status"),
                            "elapsed_seconds": result.get("elapsed_seconds"),
                            "thought_present": bool(result.get("native_thought_chars")),
                            "predicted_verdict": result.get("predicted_verdict"),
                            "correct": result.get("correct"),
                            "error_type": result.get("error_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = _build_summary(
        existing_records,
        args=args,
        manifest=manifest,
        output_dir=output_dir,
        selection_counts=selection_counts,
    )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run image-only direct QA with a shared comparison prompt."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--prompt-file")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--stratify-by",
        choices=(
            "factual_status",
            "construction_subroute",
            "target_route",
            "target_capability_cell",
            "production_class",
        ),
        help="Round-robin sample evenly across values of this manifest field.",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
