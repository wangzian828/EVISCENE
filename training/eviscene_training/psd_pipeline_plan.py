"""Resource-aware schedule for the lightweight PSD production pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    requires: tuple[str, ...]
    resource: str
    streaming: bool = False


# ``streaming`` stages consume sealed case records.  They do not wait for the
# complete rollout set, and their ledgers are append-only and restartable.
STAGES = (
    Stage("collect", (), "qwen", True),
    Stage("source_review", ("collect",), "gemini", True),
    Stage("postprocess", ("source_review",), "cpu", True),
    Stage("candidate_extract", ("postprocess",), "cpu", True),
    Stage("repair_propose", ("candidate_extract",), "gemini", True),
    Stage("repair_rollout", ("repair_propose",), "qwen", True),
    Stage("repair_check", ("repair_rollout",), "gemini", True),
    Stage("freeze_bank", ("collect", "candidate_extract", "repair_check"), "cpu"),
    Stage("teacher_topk", ("freeze_bank",), "gpu_all"),
    Stage("datum_pack", ("teacher_topk",), "cpu"),
    Stage("train_gate", ("datum_pack",), "gpu_all"),
    Stage("train_5epoch", ("train_gate",), "gpu_all"),
    Stage("verify_training", ("train_5epoch",), "gpu_all"),
)


def stage_map() -> dict[str, Stage]:
    return {stage.name: stage for stage in STAGES}


def validate_plan() -> None:
    stages = stage_map()
    if len(stages) != len(STAGES):
        raise ValueError("duplicate PSD pipeline stage")
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError("cyclic PSD pipeline plan")
        visiting.add(name)
        for dependency in stages[name].requires:
            if dependency not in stages:
                raise ValueError(f"unknown PSD dependency: {dependency}")
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in stages:
        visit(name)


def runnable(completed: set[str], running: set[str]) -> list[str]:
    """Return whole-bank stages that can start without resource collisions.

    Streaming stages are coordinated per sealed-case ledger and may overlap:
    Gemini and CPU work can run while collection owns Qwen.  Qwen repair is
    intentionally held while collection is active so it cannot slow sampling.
    """
    validate_plan()
    stages = stage_map()
    busy = {stages[name].resource for name in running}
    ready: list[str] = []
    for stage in STAGES:
        if stage.name in completed or stage.name in running:
            continue
        if not set(stage.requires).issubset(completed):
            continue
        if stage.resource in busy:
            continue
        if stage.name == "repair_rollout" and "collect" in running:
            continue
        ready.append(stage.name)
        busy.add(stage.resource)
    return ready
