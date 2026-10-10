from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, Mapping


def classification_prediction(
    sample: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Dict[str, Any] | None:
    verdict = str(result.get("verdict") or "").strip()
    if (
        verdict not in {"real", "fake"}
        or str(result.get("termination") or "") != "success"
        or bool(str(result.get("error") or "").strip())
    ):
        return None
    return {"case_id": sample.get("case_id"), "verdict": verdict}


def result_status(result: Mapping[str, Any]) -> str:
    error = str(result.get("error") or "").strip()
    termination = str(result.get("termination") or "").strip()
    if error or termination == "error" or result.get("verdict") == "error":
        return "error"
    return "success"


def run_result_record(
    sample: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    trace_path: str | None,
    metadata: Mapping[str, Any] | None,
    episode_id: str,
    prompt_group_id: str,
    rollout_index: int,
    sampling_seed: int,
) -> Dict[str, Any]:
    state = result.get("state") if isinstance(result.get("state"), Mapping) else {}
    judgment = result.get("judgment")
    verdict_basis = result.get("verdict_basis")
    if verdict_basis is None and isinstance(judgment, Mapping):
        verdict_basis = judgment.get("verdict_basis")
    fact_check_report = result.get("fact_check_report")
    evidence_citations = result.get("evidence_citations")
    if isinstance(judgment, Mapping):
        if fact_check_report is None:
            fact_check_report = judgment.get("fact_check_report")
        if evidence_citations is None:
            evidence_citations = judgment.get("evidence_citations")
    record: Dict[str, Any] = {
        "case_id": sample.get("case_id"),
        "episode_id": episode_id,
        "prompt_group_id": prompt_group_id,
        "rollout_index": rollout_index,
        "sampling_seed": sampling_seed,
        "status": result_status(result),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "verdict_basis": verdict_basis,
        "fact_check_report": fact_check_report,
        "evidence_citations": evidence_citations,
        "termination": result.get("termination"),
        "time_taken": result.get("time_taken"),
        "total_tool_calls": result.get("total_tool_calls"),
        "llm_api_calls": result.get("llm_api_calls"),
        "token_usage": result.get("token_usage"),
        "error": result.get("error"),
        "stage_timings": state.get("stage_timings", {}),
        "trace_path": trace_path,
    }
    if metadata:
        record["metadata"] = dict(metadata)
    return record


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def compute_summary(run_results: list[Dict[str, Any]]) -> Dict[str, Any]:
    verdicts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    times: list[float] = []
    tool_calls: list[float] = []
    llm_calls: list[float] = []
    for row in run_results:
        verdicts[str(row.get("verdict") or "")] += 1
        statuses[str(row.get("status") or "")] += 1
        if isinstance(row.get("time_taken"), (int, float)):
            times.append(float(row["time_taken"]))
        if isinstance(row.get("total_tool_calls"), (int, float)):
            tool_calls.append(float(row["total_tool_calls"]))
        if isinstance(row.get("llm_api_calls"), (int, float)):
            llm_calls.append(float(row["llm_api_calls"]))
    return {
        "num_episodes": len(run_results),
        "num_errors": int(statuses.get("error", 0)),
        "status_distribution": dict(statuses),
        "predicted_verdict_distribution": dict(verdicts),
        "avg_time_taken_sec": round(safe_mean(times), 3),
        "avg_tool_calls": round(safe_mean(tool_calls), 3),
        "avg_llm_api_calls": round(safe_mean(llm_calls), 3),
    }
