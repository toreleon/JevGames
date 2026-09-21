"""Stable contracts between the generic engine, model adapters, and tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DecisionOption:
    key: str
    description: str


@dataclass(frozen=True, slots=True)
class ExpertDecision:
    serialized_state: str
    observation: Mapping[str, Any]
    instruction: str
    options: tuple[DecisionOption, ...]
    target_key: str
    episode_id: str
    step: int


@dataclass(frozen=True, slots=True)
class TaskTransition:
    state: Any
    reward: float
    terminated: bool
    solved: bool
    reason: str | None = None
    primitive_steps: int = 1


@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    serialized_state: str
    instruction: str
    options: tuple[DecisionOption, ...]
    action_index: int
    old_log_probability: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    reward: float
    solved: bool
    decisions: int
    environment_steps: int
    primitive_steps: int
    reason: str
    policy_records: tuple[PolicyDecisionRecord, ...]


class DecisionTask(ABC):
    """An environment/dataset plugin; the engine treats state as opaque."""

    name: str
    instruction: str

    @abstractmethod
    def initial_states(self, dataset_path: str | Path, limit: int | None = None) -> list[Any]: ...

    @abstractmethod
    def expert_decisions(self, dataset_path: str | Path) -> list[ExpertDecision]: ...

    @abstractmethod
    def observation(self, state: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def options(self, state: Any) -> tuple[DecisionOption, ...]: ...

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

    def repeated_state_reward(self) -> float:
        return -4.0

    def no_action_reward(self) -> float:
        return -15.0

    def limit_reward(self) -> float:
        return -1.0


class DecisionModelAdapter(ABC):
    """A trainable decision model with a normalized dynamic-choice API."""

    name: str
    model: Any
    device: Any

    @abstractmethod
    def encode(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        options: Sequence[DecisionOption],
        target_index: int = 0,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def collate(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]: ...

    @abstractmethod
    def logits(self, active_model: Any, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def action_mask(self, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def labels(self, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def supervised_loss(self, active_model: Any, batch: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    def freeze_backbone(self) -> None: ...

    @abstractmethod
    def save(self, model: Any, destination: str | Path, metadata: Mapping[str, Any]) -> Path: ...

    def trainable_parameters(self) -> list[Any]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]
