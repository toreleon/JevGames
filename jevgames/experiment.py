"""Config-driven experiment orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from .config import ExperimentConfig
from .engine import encode_calibration_items, evaluate_calibration, evaluate_environment
from .registry import create_evidence, create_model, create_strategy, create_task


def run_experiment(config: ExperimentConfig) -> dict:
    started = time.perf_counter()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = create_model(config.model_type, config.model)
    task = create_task(config.task_type, config.task)
    strategy = create_strategy(config.training_type, config.training)
    evidence_provider = create_evidence(config.evidence_type, config.evidence)

    evidence = evidence_provider.load(config.train_dataset, task)
    train_items = encode_calibration_items(adapter, evidence)
    stats = [{
        "phase": "setup",
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "training_strategy": config.training_type,
        "evidence_plugin": config.evidence_type,
        "calibration_evidence": len(evidence),
        "trainable_decisions": len(train_items),
        "provenance": dict(sorted(Counter(row.provenance for row in evidence).items())),
    }]
    stats.extend(strategy.train(adapter, train_items, output, config.seed))

    checkpoint_dir = output / "checkpoint"
    adapter.save(adapter.model, checkpoint_dir, {
        "experiment": config.name,
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "training_strategy": config.training_type,
        "evidence_plugin": config.evidence_type,
        "stats": stats,
    })

    benchmarks = {}
    for split, dataset_path in (
        ("validation", config.validation_dataset),
        ("test", config.test_dataset),
    ):
        if not dataset_path:
            continue
        split_evidence = evidence_provider.load(dataset_path, task)
        split_items = encode_calibration_items(adapter, split_evidence)
        benchmarks[split] = {
            "calibration": evaluate_calibration(adapter, split_items, config.benchmark),
            "environment": evaluate_environment(
                adapter,
                task,
                task.initial_states(dataset_path, config.benchmark.max_instances),
                config.benchmark,
            ),
        }

    report = {
        "schema_version": 2,
        "name": config.name,
        "config": asdict(config),
        "plugins": {
            "model": config.model_type,
            "evidence": config.evidence_type,
            "strategy": config.training_type,
            "task": config.task_type,
        },
        "stats": stats,
        "benchmarks": benchmarks,
        "checkpoint": str(checkpoint_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
