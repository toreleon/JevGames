"""Stochastic environment collection with terminal-outcome labels."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Any, Mapping, Sequence

from jevgames.contracts import (
    ActionOutcomeQuery,
    CalibrationDecision,
    CollectedEvidence,
    DecisionModelAdapter,
    DecisionTask,
    EvidenceCollector,
)
from jevgames.registry import register_collector


@dataclass(frozen=True, slots=True)
class EpisodicCollectorConfig:
    iterations: int = 2
    episodes_per_instance: int = 4
    max_instances: int | None = None
    max_decisions: int = 32
    inference_batch_size: int = 64
    sampling_temperature: float = 1.0
    epsilon: float = 0.10

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.episodes_per_instance < 1:
            raise ValueError("collection iteration and episode counts must be valid")
        if self.max_instances is not None and self.max_instances < 1:
            raise ValueError("collection.max_instances must be positive when set")
        if self.max_decisions < 1 or self.inference_batch_size < 1:
            raise ValueError("collection decision and batch counts must be positive")
        if self.sampling_temperature <= 0.0:
            raise ValueError("collection.sampling_temperature must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("collection.epsilon must be between zero and one")


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    serialized_state: str
    observation: Mapping[str, Any]
    question: Any
    step: int


@dataclass(slots=True)
class _LiveEpisode:
    state: Any
    seen: set[Any]
    decisions: list[_PendingDecision]
    environment_steps: int = 0
    reason: str | None = None


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _sample_query(
    queries: Sequence[tuple[ActionOutcomeQuery, float]],
    temperature: float,
    epsilon: float,
    rng: random.Random,
) -> ActionOutcomeQuery:
    logits = [math.log(max(probability, 1e-8)) / temperature for _query, probability in queries]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    count = len(weights)
    weights = [(1.0 - epsilon) * value / total + epsilon / count for value in weights]
    return rng.choices([query for query, _probability in queries], weights=weights, k=1)[0]


class EpisodicOutcomeCollector(EvidenceCollector):
    name = "episodic_outcomes"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = EpisodicCollectorConfig(**config)
        self.iterations = self.config.iterations
        self.max_instances = self.config.max_instances

    def collect(
        self,
        adapter: DecisionModelAdapter,
        task: DecisionTask,
        initial_states: Sequence[Any],
        seed: int,
        iteration: int,
    ) -> CollectedEvidence:
        import torch

        selected_states = list(
            initial_states[: self.config.max_instances]
            if self.config.max_instances is not None
            else initial_states
        )
        if not selected_states:
            raise ValueError("episodic collection has no initial states")
        rng = random.Random(seed + iteration * 1_000_003)
        live = [
            _LiveEpisode(state=state, seen=set(), decisions=[])
            for state in selected_states
            for _repeat in range(self.config.episodes_per_instance)
        ]
        adapter.model.eval()

        for _round in range(self.config.max_decisions):
            requests: list[tuple[int, ActionOutcomeQuery, dict[str, Any]]] = []
            for episode_index, episode in enumerate(live):
                if episode.reason is not None:
                    continue
                if task.is_solved(episode.state):
                    episode.reason = "solved"
                    continue
                state_key = task.state_key(episode.state)
                if state_key in episode.seen:
                    episode.reason = "repeat"
                    continue
                episode.seen.add(state_key)
                queries = tuple(
                    query for query in task.outcome_queries(episode.state)
                    if not task.is_known_terminal_failure(episode.state, query.action_key)
                )
                if not queries:
                    episode.reason = "no_action"
                    continue
                observation = task.observation(episode.state)
                for query in queries:
                    requests.append((
                        episode_index,
                        query,
                        adapter.encode(observation, query.question),
                    ))
            if not requests:
                break

            scored: dict[int, list[tuple[ActionOutcomeQuery, float]]] = {}
            for offset in range(0, len(requests), self.config.inference_batch_size):
                chunk = requests[offset : offset + self.config.inference_batch_size]
                batch = _to_device(adapter.collate([request[2] for request in chunk]), adapter.device)
                with torch.no_grad():
                    logits = adapter.logits(adapter.model, batch)
                    mask = adapter.action_mask(batch).bool()
                    probabilities = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
                for row, (episode_index, query, _item) in enumerate(chunk):
                    scored.setdefault(episode_index, []).append((query, float(probabilities[row, 1].cpu())))

            for episode_index, candidates in scored.items():
                episode = live[episode_index]
                selected = _sample_query(
                    candidates,
                    self.config.sampling_temperature,
                    self.config.epsilon,
                    rng,
                )
                episode.decisions.append(_PendingDecision(
                    serialized_state=task.serialize_state(episode.state),
                    observation=task.observation(episode.state),
                    question=selected.question,
                    step=episode.environment_steps,
                ))
                outcome = task.transition(episode.state, selected.action_key)
                episode.state = outcome.state
                episode.environment_steps += 1
                if outcome.terminated:
                    episode.reason = outcome.reason or ("solved" if outcome.solved else "terminated")

        for episode in live:
            if episode.reason is None:
                episode.reason = "decision_limit"

        decisions = []
        positive_decisions = 0
        negative_decisions = 0
        unknown_decisions = 0
        for episode_index, episode in enumerate(live):
            solved = task.is_solved(episode.state)
            labeled: list[tuple[_PendingDecision, tuple[float, float]]] = []
            if solved:
                labeled.extend((pending, (0.0, 1.0)) for pending in episode.decisions)
                positive_decisions += len(episode.decisions)
            elif episode.reason in {"static_deadlock", "no_action"} and episode.decisions:
                labeled.append((episode.decisions[-1], (1.0, 0.0)))
                negative_decisions += 1
                unknown_decisions += len(episode.decisions) - 1
            else:
                unknown_decisions += len(episode.decisions)
            for pending, target in labeled:
                decisions.append(CalibrationDecision(
                    serialized_state=pending.serialized_state,
                    observation=pending.observation,
                    question=pending.question,
                    target_probabilities=target,
                    evidence_id=f"collection_{iteration:03d}_{episode_index:08d}",
                    step=pending.step,
                    provenance=f"online_episode_outcome:iteration_{iteration}",
                ))

        outcomes = {
            reason: sum(episode.reason == reason for episode in live)
            for reason in sorted({episode.reason for episode in live})
        }
        stats = {
            "phase": "environment_collection",
            "collector": self.name,
            "iteration": iteration,
            "instances": len(selected_states),
            "episodes": len(live),
            "solved": sum(task.is_solved(episode.state) for episode in live),
            "decisions": len(decisions),
            "positive_decisions": positive_decisions,
            "negative_decisions": negative_decisions,
            "unknown_decisions": unknown_decisions,
            "mean_environment_steps": round(fmean(episode.environment_steps for episode in live), 6),
            "sampling_temperature": self.config.sampling_temperature,
            "epsilon": self.config.epsilon,
            "outcomes": outcomes,
        }
        return CollectedEvidence(tuple(decisions), stats)


@register_collector("episodic_outcomes")
def create_episodic_outcome_collector(config: dict[str, Any]) -> EpisodicOutcomeCollector:
    return EpisodicOutcomeCollector(config)
