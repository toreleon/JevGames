"""Config-driven experiment orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from .config import ExperimentConfig
from .engine import encode_calibration_items, evaluate_calibration, evaluate_environment
from .registry import create_collector, create_evidence, create_model, create_strategy, create_task


def run_experiment(config: ExperimentConfig) -> dict:
    started = time.perf_counter()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_config = dict(config.model)
    model_config.setdefault("seed", config.seed)
    adapter = create_model(config.model_type, model_config)
    task = create_task(config.task_type, config.task)
    strategy = create_strategy(config.training_type, config.training)
    evidence_provider = create_evidence(config.evidence_type, config.evidence)
    collector = create_collector(config.collector_type, config.collector) if config.collector_type else None

    evidence = evidence_provider.load(config.train_dataset, task)
    train_items = encode_calibration_items(adapter, evidence)
    stats = [{
        "phase": "setup",
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "training_strategy": config.training_type,
        "evidence_plugin": config.evidence_type,
        "collector_plugin": config.collector_type,
        "calibration_evidence": len(evidence),
        "trainable_decisions": len(train_items),
        "provenance": dict(sorted(Counter(row.provenance for row in evidence).items())),
    }]
    stats.extend(strategy.train(adapter, train_items, output, config.seed, stage="initial"))

    if collector is not None:
        initial_states = task.initial_states(config.train_dataset, collector.max_instances)
        for iteration in range(1, collector.iterations + 1):
            collected = collector.collect(
                adapter,
                task,
                initial_states,
                config.seed,
                iteration,
            )
            stats.append(dict(collected.stats))
            print(json.dumps(collected.stats, sort_keys=True), flush=True)
            if not collected.decisions:
                continue
            replay_evidence = list(evidence) + list(collected.decisions)
            replay_items = encode_calibration_items(adapter, replay_evidence)
            stats.extend(strategy.train(
                adapter,
                replay_items,
                output,
                config.seed + iteration,
                stage=f"collection_{iteration}",
            ))

    if config.calibration_dataset:
        calibration_evidence = evidence_provider.load(config.calibration_dataset, task)
        if collector is not None:
            calibration_collection = collector.collect(
                adapter,
                task,
                task.initial_states(config.calibration_dataset, collector.max_instances),
                config.seed + 10_000_019,
                collector.iterations + 1,
            )
            calibration_evidence.extend(calibration_collection.decisions)
            calibration_collection_stat = dict(calibration_collection.stats)
            calibration_collection_stat["phase"] = "calibration_collection"
            stats.append(calibration_collection_stat)
            print(json.dumps(calibration_collection_stat, sort_keys=True), flush=True)
        calibration_items = encode_calibration_items(adapter, calibration_evidence)
        calibration_stat = dict(adapter.fit_calibration(calibration_items, config.benchmark.batch_size))
        calibration_stat.update({
            "phase": "post_training_calibration",
            "evidence": len(calibration_items),
        })
        stats.append(calibration_stat)
        print(json.dumps(calibration_stat, sort_keys=True), flush=True)

    checkpoint_dir = output / "checkpoint"
    adapter.save(adapter.model, checkpoint_dir, {
        "experiment": config.name,
        "model_plugin": config.model_type,
        "task_plugin": config.task_type,
        "training_strategy": config.training_type,
        "evidence_plugin": config.evidence_type,
        "collector_plugin": config.collector_type,
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
        split_report = {
            "calibration": evaluate_calibration(adapter, split_items, config.benchmark),
            "environment": evaluate_environment(
                adapter,
                task,
                task.initial_states(dataset_path, config.benchmark.max_instances),
                config.benchmark,
            ),
        }
        if collector is not None:
            outcome_benchmark = collector.collect(
                adapter,
                task,
                task.initial_states(dataset_path, collector.max_instances),
                config.seed + (20_000_033 if split == "validation" else 30_000_061),
                collector.iterations + (2 if split == "validation" else 3),
            )
            split_report["outcome_collection"] = dict(outcome_benchmark.stats)
            if outcome_benchmark.decisions:
                split_report["outcome_calibration"] = evaluate_calibration(
                    adapter,
                    encode_calibration_items(adapter, outcome_benchmark.decisions),
                    config.benchmark,
                )
        benchmarks[split] = split_report

    report = {
        "schema_version": 4,
        "name": config.name,
        "config": asdict(config),
        "plugins": {
            "model": config.model_type,
            "collector": config.collector_type,
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
