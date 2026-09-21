"""Stable contracts between training strategies, model adapters, and tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DecisionOption:
    key: str
    description: str


class DecisionKind(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True, slots=True)
class DecisionQuestion:
    key: str
    kind: DecisionKind
    instruction: str
    options: tuple[DecisionOption, ...]

    def __post_init__(self) -> None:
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("decision option keys must be unique")
        if self.kind in {DecisionKind.SCORE, DecisionKind.NOUL} and len(keys) < 2:
            raise ValueError(f"{self.kind.value} decisions require at least two options")
        if self.kind is DecisionKind.NOUL and tuple(keys) != ("false", "true"):
            raise ValueError("noul decisions require options ordered as false, true")


@dataclass(frozen=True, slots=True)
class ActionOutcomeQuery:
    """A binary outcome question associated with one executable action."""

    action_key: str
    question: DecisionQuestion

    def __post_init__(self) -> None:
        if self.question.kind is not DecisionKind.NOUL:
            raise ValueError("action outcome queries must use noul questions")


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    """One typed decision and its observed target distribution.

    A one-hot distribution represents one observed outcome. Soft targets are
    also supported, allowing a task to aggregate repeated solver or
    environment outcomes without changing the training strategy.
    """

    serialized_state: str
    observation: Mapping[str, Any]
    question: DecisionQuestion
    target_probabilities: tuple[float, ...]
    evidence_id: str
    step: int
    provenance: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.question.options) != len(self.target_probabilities):
            raise ValueError("target_probabilities must match the decision options")
        if any(not isfinite(value) or value < 0.0 for value in self.target_probabilities):
            raise ValueError("target probabilities must be finite and non-negative")
        if abs(sum(self.target_probabilities) - 1.0) > 1e-6:
            raise ValueError("target probabilities must sum to one")

    @classmethod
    def one_hot(
        cls,
        *,
        serialized_state: str,
        observation: Mapping[str, Any],
        question: DecisionQuestion,
        target_key: str,
        evidence_id: str,
        step: int,
        provenance: str,
    ) -> "CalibrationDecision":
        keys = tuple(option.key for option in question.options)
        if target_key not in keys:
            raise ValueError(f"target {target_key!r} is absent from the decision options")
        target = tuple(1.0 if key == target_key else 0.0 for key in keys)
        return cls(
            serialized_state=serialized_state,
            observation=observation,
            question=question,
            target_probabilities=target,
            evidence_id=evidence_id,
            step=step,
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class TaskTransition:
    state: Any
    reward: float
    terminated: bool
    solved: bool
    reason: str | None = None
    primitive_steps: int = 1


class DecisionTask(ABC):
    """A task plugin that defines a typed-decision environment."""

    name: str
    @abstractmethod
    def initial_states(self, dataset_path: str | Path, limit: int | None = None) -> list[Any]: ...

    @abstractmethod
    def observation(self, state: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def question(self, state: Any) -> DecisionQuestion: ...

    def outcome_queries(self, state: Any) -> tuple[ActionOutcomeQuery, ...]:
        """Return per-action terminal-outcome questions when the task supports them."""

        return ()

    @abstractmethod
    def transition(self, state: Any, option_key: str) -> TaskTransition: ...

    @abstractmethod
    def serialize_state(self, state: Any) -> str: ...

    @abstractmethod
    def deserialize_state(self, value: str) -> Any: ...

    @abstractmethod
    def state_key(self, state: Any) -> Hashable: ...

    @abstractmethod
    def is_solved(self, state: Any) -> bool: ...


class EvidenceProvider(ABC):
    """A dataset adapter that creates calibration evidence for a task."""

    name: str

    @abstractmethod
    def load(
        self,
        dataset_path: str | Path,
        task: DecisionTask,
    ) -> list[CalibrationDecision]: ...


class DecisionModelAdapter(ABC):
    """A trainable typed-decision model with dynamic option sets."""

    name: str
    model: Any
    device: Any

    @abstractmethod
    def encode(
        self,
        observation: Mapping[str, Any],
        question: DecisionQuestion,
        target_probabilities: Sequence[float] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def collate(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]: ...

    @abstractmethod
    def logits(self, active_model: Any, batch: Mapping[str, Any]) -> Any: ...

    def training_logits(self, active_model: Any, batch: Mapping[str, Any]) -> Any:
        """Return logits before any post-training calibration transform."""

        return self.logits(active_model, batch)

    @abstractmethod
    def action_mask(self, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def target_probabilities(self, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def ordinal_mask(self, batch: Mapping[str, Any]) -> Any:
        """Mark evidence rows whose ordered options should receive RPS."""

    @abstractmethod
    def freeze_backbone(self) -> None: ...

    @abstractmethod
    def save(self, model: Any, destination: str | Path, metadata: Mapping[str, Any]) -> Path: ...

    def trainable_parameters(self) -> list[Any]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def fit_calibration(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int,
    ) -> Mapping[str, Any]:
        raise NotImplementedError(f"model adapter {self.name!r} does not support post-training calibration")


class TrainingStrategy(ABC):
    """A pluggable optimizer over encoded calibration evidence."""

    name: str

    @abstractmethod
    def train(
        self,
        adapter: DecisionModelAdapter,
        items: Sequence[dict[str, Any]],
        output_dir: str | Path,
        seed: int,
        stage: str = "initial",
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class CollectedEvidence:
    decisions: tuple[CalibrationDecision, ...]
    stats: Mapping[str, Any]


class EvidenceCollector(ABC):
    """Collect labeled decision evidence by interacting with an environment."""

    name: str
    iterations: int
    max_instances: int | None

    @abstractmethod
    def collect(
        self,
        adapter: DecisionModelAdapter,
        task: DecisionTask,
        initial_states: Sequence[Any],
        seed: int,
        iteration: int,
    ) -> CollectedEvidence: ...
