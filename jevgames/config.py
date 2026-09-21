"""TOML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    max_decisions: int = 32
    batch_size: int = 32
    max_instances: int | None = None
    calibration_bins: int = 15

    def __post_init__(self) -> None:
        if self.max_decisions < 1 or self.batch_size < 1 or self.calibration_bins < 1:
            raise ValueError("benchmark decision, batch, and bin counts must be positive")
        if self.max_instances is not None and self.max_instances < 1:
            raise ValueError("benchmark.max_instances must be positive when set")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int
    model_type: str
    model: dict[str, Any]
    task_type: str
    task: dict[str, Any]
    training_type: str
    training: dict[str, Any]
    evidence_type: str
    evidence: dict[str, Any]
    collector_type: str | None
    collector: dict[str, Any]
    train_dataset: str
    calibration_dataset: str | None = None
    validation_dataset: str | None = None
    test_dataset: str | None = None
    manifest: str | None = None
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def _construct(cls, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    legacy = sorted(section for section in ("warmup", "online") if section in raw)
    if legacy:
        names = ", ".join(f"[{section}]" for section in legacy)
        raise ValueError(f"legacy training sections {names} are unsupported; configure [training] type = 'rlcd'")

    experiment = raw.get("experiment", {})
    model = raw.get("model", {})
    task = raw.get("task", {})
    training = raw.get("training", {})
    collection = raw.get("collection", {})
    data = raw.get("data", {})
    if "type" not in model or "type" not in task or "type" not in training or "type" not in data:
        raise ValueError("config requires model.type, task.type, training.type, and data.type")
    if "train" not in data:
        raise ValueError("config requires data.train")
    return ExperimentConfig(
        name=str(experiment.get("name", source.stem)),
        output_dir=str(experiment.get("output_dir", f"runs/{source.stem}")),
        seed=int(experiment.get("seed", 42)),
        model_type=str(model["type"]),
        model={key: value for key, value in model.items() if key != "type"},
        task_type=str(task["type"]),
        task={key: value for key, value in task.items() if key != "type"},
        training_type=str(training["type"]),
        training={key: value for key, value in training.items() if key != "type"},
        evidence_type=str(data["type"]),
        evidence={
            key: value
            for key, value in data.items()
            if key not in {"type", "train", "calibration", "validation", "test", "manifest"}
        },
        collector_type=str(collection["type"]) if "type" in collection else None,
        collector={key: value for key, value in collection.items() if key != "type"},
        train_dataset=str(data["train"]),
        calibration_dataset=data.get("calibration"),
        validation_dataset=data.get("validation"),
        test_dataset=data.get("test"),
        manifest=data.get("manifest"),
        benchmark=_construct(BenchmarkConfig, raw.get("benchmark")),
    )
