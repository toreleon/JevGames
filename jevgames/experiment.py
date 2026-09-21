"""Config-driven experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from .config import ExperimentConfig
from .engine import _expert_items, evaluate, online_grpo, supervised_warmup
from .registry import create_model, create_task


def run_experiment(config: ExperimentConfig) -> dict:
    started = time.perf_counter()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = create_model(config.model_type, config.model)
    task = create_task(config.task_type, config.task)
    experts = task.expert_decisions(config.train_dataset)
    expert_items = _expert_items(adapter, experts)
    stats = [{
        "phase": "setup",
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "expert_decisions": len(experts),
        "trainable_expert_decisions": len(expert_items),
    }]

    if config.warmup.epochs > 0:
        warmup = supervised_warmup(adapter, expert_items, config.warmup, output, config.seed)
        stats.append(warmup)
        print(json.dumps(warmup, sort_keys=True), flush=True)
    train_states = task.initial_states(config.train_dataset, config.online.max_instances)
    if config.online.iterations > 0:
        stats.extend(online_grpo(adapter, task, expert_items, train_states, config.online, config.seed))

    checkpoint_dir = output / "checkpoint"
    adapter.save(adapter.model, checkpoint_dir, {
        "experiment": config.name,
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "stats": stats,
    })
    benchmarks = {}
    if config.validation_dataset:
        benchmarks["validation"] = evaluate(
            adapter,
            task,
            task.initial_states(config.validation_dataset, config.benchmark.max_instances),
            config.benchmark,
        )
    if config.test_dataset:
        benchmarks["test"] = evaluate(
            adapter,
            task,
            task.initial_states(config.test_dataset, config.benchmark.max_instances),
            config.benchmark,
        )
    report = {
        "schema_version": 1,
        "name": config.name,
        "config": asdict(config),
        "plugins": {"model": config.model_type, "task": config.task_type},
        "stats": stats,
        "benchmarks": benchmarks,
        "checkpoint": str(checkpoint_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
