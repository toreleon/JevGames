"""TOML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    epochs: int = 6
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 25


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    iterations: int = 2
    group_size: int = 4
    max_decisions: int = 32
    max_instances: int | None = None
    batch_size: int = 16
    learning_rate: float = 3e-4
    gradient_accumulation: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    rollout_instance_batch_size: int = 16
    rollout_inference_batch_size: int = 32
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    expert_weight: float = 0.4


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    max_decisions: int = 32
    batch_size: int = 32
    max_instances: int | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int
    model_type: str
    model: dict[str, Any]
    task_type: str
    task: dict[str, Any]
    train_dataset: str
    validation_dataset: str | None = None
    test_dataset: str | None = None
    manifest: str | None = None
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def _construct(cls, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    experiment = raw.get("experiment", {})
    model = raw.get("model", {})
    task = raw.get("task", {})
    data = raw.get("data", {})
    if "type" not in model or "type" not in task:
        raise ValueError("config requires model.type and task.type")
    return ExperimentConfig(
        name=str(experiment.get("name", source.stem)),
        output_dir=str(experiment.get("output_dir", f"runs/{source.stem}")),
        seed=int(experiment.get("seed", 42)),
        model_type=str(model["type"]),
        model={key: value for key, value in model.items() if key != "type"},
        task_type=str(task["type"]),
        task={key: value for key, value in task.items() if key != "type"},
        train_dataset=str(data["train"]),
        validation_dataset=data.get("validation"),
        test_dataset=data.get("test"),
        manifest=data.get("manifest"),
        warmup=_construct(WarmupConfig, raw.get("warmup")),
        online=_construct(OnlineConfig, raw.get("online")),
        benchmark=_construct(BenchmarkConfig, raw.get("benchmark")),
    )
