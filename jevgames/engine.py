"""Framework-owned evidence encoding and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from .config import BenchmarkConfig
from .contracts import CalibrationDecision, DecisionModelAdapter, DecisionOption, DecisionTask


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def encode_calibration_items(
    adapter: DecisionModelAdapter,
    evidence: Sequence[CalibrationDecision],
) -> list[dict[str, Any]]:
    items = []
    for decision in evidence:
        if len(decision.options) < 2:
            continue
        items.append(adapter.encode(
            decision.observation,
            decision.instruction,
            decision.options,
            decision.target_probabilities,
        ))
    if not items:
        raise ValueError("dataset has no calibration decisions with at least two options")
    return items


def _reliability_bins(
    confidences: Sequence[float],
    correctness: Sequence[float],
    count: int,
) -> tuple[float, list[dict[str, Any]]]:
    if count < 1:
        raise ValueError("calibration bin count must be positive")
    rows = []
    weighted_gap = 0.0
    total = max(1, len(confidences))
    for index in range(count):
        lower = index / count
        upper = (index + 1) / count
        selected = [
            row for row, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == count - 1 and confidence == 1.0)
        ]
        if not selected:
            continue
        mean_confidence = fmean(confidences[row] for row in selected)
        accuracy = fmean(correctness[row] for row in selected)
        gap = abs(mean_confidence - accuracy)
        weighted_gap += len(selected) / total * gap
        rows.append({
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "count": len(selected),
            "mean_confidence": round(mean_confidence, 6),
            "accuracy": round(accuracy, 6),
            "gap": round(gap, 6),
        })
    return weighted_gap, rows


def evaluate_calibration(
    adapter: DecisionModelAdapter,
    items: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Measure decision quality and probability calibration on held-out evidence."""

    import torch

    selected = list(items[: config.max_instances] if config.max_instances else items)
    if not selected:
        raise ValueError("calibration benchmark has no evidence")
    confidences: list[float] = []
    correctness: list[float] = []
    soft_correctness: list[float] = []
    negative_log_likelihoods: list[float] = []
    brier_scores: list[float] = []
    proper_scores: list[float] = []
    adapter.model.eval()
    for offset in range(0, len(selected), config.batch_size):
        batch = _to_device(adapter.collate(selected[offset : offset + config.batch_size]), adapter.device)
        with torch.no_grad():
            logits = adapter.logits(adapter.model, batch)
            mask = adapter.action_mask(batch).bool()
            probabilities = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
            targets = adapter.target_probabilities(batch)
            rewards = adapter.calibration_reward(probabilities, batch)
        for row in range(int(probabilities.shape[0])):
            valid = int(mask[row].sum().item())
            predicted = int(probabilities[row, :valid].argmax().item())
            expected = int(targets[row, :valid].argmax().item())
            probability = probabilities[row, :valid].float()
            target = targets[row, :valid].float()
            confidences.append(float(probability[predicted].item()))
            correctness.append(float(predicted == expected))
            soft_correctness.append(float(target[predicted].item()))
            negative_log_likelihoods.append(float(-(target * probability.clamp_min(1e-12).log()).sum().item()))
            brier_scores.append(float(((probability - target) ** 2).sum().item()))
            proper_scores.append(float(rewards[row].item()))
    ece, reliability = _reliability_bins(confidences, correctness, config.calibration_bins)
    return {
        "instances": len(selected),
        "accuracy": round(fmean(correctness), 6),
        "soft_accuracy": round(fmean(soft_correctness), 6),
        "negative_log_likelihood": round(fmean(negative_log_likelihoods), 6),
        "brier_score": round(fmean(brier_scores), 6),
        "expected_calibration_error": round(ece, 6),
        "mean_confidence": round(fmean(confidences), 6),
        "mean_proper_score": round(fmean(proper_scores), 6),
        "reliability_bins": reliability,
    }


@dataclass(slots=True)
class _LiveEpisode:
    state: Any
    seen: set[Any]
    environment_steps: int = 0
    primitive_steps: int = 0
    reason: str | None = None


def evaluate_environment(
    adapter: DecisionModelAdapter,
    task: DecisionTask,
    states: Sequence[Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run greedy model-only play; this benchmark never updates the model."""

    import torch

    selected_states = list(states[: config.max_instances] if config.max_instances else states)
    if not selected_states:
        raise ValueError("environment benchmark has no initial states")
    live = [_LiveEpisode(state, set()) for state in selected_states]
    adapter.model.eval()
    for _round in range(config.max_decisions):
        selections: dict[int, DecisionOption] = {}
        requests = []
        for index, episode in enumerate(live):
            if episode.reason is not None:
                continue
            if task.is_solved(episode.state):
                episode.reason = "solved"
                continue
            key = task.state_key(episode.state)
            if key in episode.seen:
                episode.reason = "repeat"
                continue
            episode.seen.add(key)
            options = task.options(episode.state)
            if not options:
                episode.reason = "no_action"
            elif len(options) == 1:
                selections[index] = options[0]
            else:
                requests.append((
                    index,
                    options,
                    adapter.encode(task.observation(episode.state), task.instruction, options),
                ))
        for offset in range(0, len(requests), config.batch_size):
            chunk = requests[offset : offset + config.batch_size]
            batch = _to_device(adapter.collate([request[2] for request in chunk]), adapter.device)
            with torch.no_grad():
                logits = adapter.logits(adapter.model, batch)
                mask = adapter.action_mask(batch).bool()
                actions = logits.masked_fill(~mask, -1e4).argmax(dim=-1)
            for row, (index, options, _item) in enumerate(chunk):
                selections[index] = options[int(actions[row].cpu())]
        if not selections:
            break
        for index, option in selections.items():
            episode = live[index]
            outcome = task.transition(episode.state, option.key)
            episode.state = outcome.state
            episode.environment_steps += 1
            episode.primitive_steps += outcome.primitive_steps
            if outcome.terminated:
                episode.reason = outcome.reason or "terminated"
    for episode in live:
        if episode.reason is None:
            episode.reason = "decision_limit"
    solved = sum(task.is_solved(episode.state) for episode in live)
    outcomes = {
        reason: sum(episode.reason == reason for episode in live)
        for reason in sorted({episode.reason for episode in live})
    }
    return {
        "instances": len(live),
        "solved": solved,
        "solve_rate": round(solved / max(1, len(live)), 6),
        "mean_environment_steps": round(fmean(episode.environment_steps for episode in live), 3),
        "mean_primitive_steps": round(fmean(episode.primitive_steps for episode in live), 3),
        "outcomes": outcomes,
    }
