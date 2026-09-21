"""Legacy primitive-action environment GRPO baseline.

The production path uses push macros and framework integrations in
``macro_grpo_training.py``.  This implementation is retained for ablations
that compare primitive movement against macro actions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean, pstdev
from typing import Any, Sequence

from .core import ACTION_DESCRIPTIONS, Action, Board
from .checkpoint import load_checkpoint, save_checkpoint
from .device import resolve_device
from .laya_policy import BOARD_LEGEND, question_for
from .training import _build_items, _collate
from .trajectories import TrainingExample, read_jsonl


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    iterations: int = 1
    group_size: int = 4
    max_steps: int = 32
    max_levels: int = 4
    batch_size: int = 4
    learning_rate: float = 5e-5
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    expert_weight: float = 0.30
    gradient_accumulation: int = 2
    freeze_encoder: bool = True
    seed: int = 7
    solve_reward: float = 50.0
    goal_reward: float = 3.0
    off_goal_penalty: float = 3.0
    push_reward: float = 0.05
    step_penalty: float = 0.02
    deadlock_penalty: float = 15.0
    repeat_penalty: float = 3.0
    limit_penalty: float = 1.0


@dataclass(frozen=True, slots=True)
class RolloutStep:
    board: str
    legal_actions: tuple[str, ...]
    action_index: int
    old_log_probability: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    reward: float
    solved: bool
    pushes: int
    steps: int
    reason: str
    decisions: tuple[RolloutStep, ...]


def _internal_question(legal: Sequence[Action]) -> dict[str, Any]:
    external = question_for(legal)
    return {"t": "choice", "ins": external["instructions"], "crit": external["criteria"]}


def _encode_policy_item(board: Board, legal: Sequence[Action], tokenizer: Any, model_cfg: dict[str, Any], label: int = 0) -> dict[str, Any]:
    from laya.common import QTYPES, build_sequence

    question = _internal_question(legal)
    ids, markers = build_sequence(
        tokenizer,
        {"board": board.render(), "legend": BOARD_LEGEND},
        question,
        model_cfg["max_len"],
        model_cfg["head_max_len"],
    )
    if len(markers) != len(legal):
        raise ValueError("Sokoban action question exceeded Laya's head token budget")
    target = [0.0] * len(legal)
    target[label] = 1.0
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target, "label": label}


def _temperature_scaled(logits: Any, mask: Any, model_cfg: dict[str, Any], torch: Any) -> Any:
    from laya.common import QTYPES, temp_bucket

    defaults = model_cfg.get("temperature", [1.0, 1.0, 1.0])
    by_options = model_cfg.get("temperature_by_options", {})
    scales = []
    for count in mask.sum(-1).tolist():
        value = by_options.get(temp_bucket(QTYPES["choice"], int(count)), defaults[QTYPES["choice"]])
        scales.append(max(1e-3, float(value)))
    return logits / torch.tensor(scales, device=logits.device, dtype=logits.dtype).unsqueeze(-1)


def initial_boards(examples: Sequence[TrainingExample], limit: int | None = None) -> list[Board]:
    """Recover one initial board per solver episode from a JSONL dataset."""

    first: dict[str, TrainingExample] = {}
    for example in examples:
        if example.step == 0 and example.episode not in first:
            first[example.episode] = example
    boards = [Board.from_ascii(example.board) for example in first.values()]
    if not boards:
        raise ValueError("expert dataset has no episode step 0 records")
    return boards[:limit] if limit is not None else boards


def _rollout_episode(model: Any, tokenizer: Any, model_cfg: dict[str, Any], initial: Board, config: GRPOConfig, rng: random.Random, torch: Any, device: Any) -> EpisodeResult:
    board = initial
    seen: set[Board] = set()
    decisions: list[RolloutStep] = []
    reward = 0.0
    pushes = 0
    reason = "step_limit"
    model.eval()

    for step_index in range(config.max_steps):
        if board.is_solved():
            reason = "solved"
            break
        if board in seen:
            reward -= config.repeat_penalty
            reason = "repeat"
            break
        seen.add(board)
        legal = board.legal_actions()
        if not legal:
            reward -= config.deadlock_penalty
            reason = "no_legal_action"
            break

        if len(legal) == 1:
            action = legal[0]
        else:
            item = _encode_policy_item(board, legal, tokenizer, model_cfg)
            batch = _collate([item], tokenizer.pad_token_id, torch)
            mask = batch["marker_mask"].to(device)
            with torch.no_grad():
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    mask,
                    batch["qtype"].to(device),
                    detach_encoder=config.freeze_encoder,
                )
                scaled = _temperature_scaled(logits.float(), mask, model_cfg, torch)
                probabilities = torch.softmax(scaled.masked_fill(~mask, -1e4), dim=-1)[0, : len(legal)].cpu().tolist()
            action_index = rng.choices(range(len(legal)), weights=probabilities, k=1)[0]
            action = legal[action_index]
            decisions.append(
                RolloutStep(
                    board=board.render(),
                    legal_actions=tuple(candidate.value for candidate in legal),
                    action_index=action_index,
                    old_log_probability=math.log(max(1e-8, probabilities[action_index])),
                )
            )

        before_goals = len(board.boxes & board.goals)
        transition = board.apply(action)
        if not transition.moved:
            raise RuntimeError("rollout selected an invalid action despite legal-action masking")
        board = transition.board
        after_goals = len(board.boxes & board.goals)
        reward -= config.step_penalty
        if transition.pushed:
            pushes += 1
            reward += config.push_reward
        delta = after_goals - before_goals
        reward += max(0, delta) * config.goal_reward
        reward -= max(0, -delta) * config.off_goal_penalty
        if board.is_solved():
            reward += config.solve_reward
            reason = "solved"
            step_index += 1
            break
        if board.has_static_deadlock():
            reward -= config.deadlock_penalty
            reason = "static_deadlock"
            step_index += 1
            break
    else:
        reward -= config.limit_penalty
        step_index = config.max_steps

    return EpisodeResult(
        reward=round(reward, 6),
        solved=board.is_solved(),
        pushes=pushes,
        steps=int(step_index),
        reason=reason,
        decisions=tuple(decisions),
    )


def _group_training_rows(episodes: Sequence[EpisodeResult]) -> list[tuple[RolloutStep, float]]:
    rewards = [episode.reward for episode in episodes]
    mean = fmean(rewards)
    deviation = pstdev(rewards)
    if deviation < 1e-6:
        return []
    rows: list[tuple[RolloutStep, float]] = []
    for episode in episodes:
        advantage = (episode.reward - mean) / (deviation + 1e-6)
        rows.extend((decision, advantage) for decision in episode.decisions)
    return rows


def train_environment_grpo(
    model_id_or_path: str | Path,
    expert_dataset: str | Path,
    output_dir: str | Path,
    config: GRPOConfig = GRPOConfig(),
    device: str = "auto",
) -> Path:
    """Run hybrid expert-replay + episodic GRPO and save a Laya checkpoint."""

    try:
        import torch
        from laya.common import proper_reward
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Install model dependencies with: uv sync --extra laya") from exc

    runtime_device = resolve_device(device)
    model, tokenizer, model_cfg = load_checkpoint(model_id_or_path, runtime_device)
    examples = read_jsonl(expert_dataset)
    expert_items = _build_items(examples, tokenizer, model_cfg)
    boards = initial_boards(examples, config.max_levels)
    rng = random.Random(config.seed)

    if config.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime_device.type == "cuda" else None
    stats: list[dict[str, Any]] = []

    for iteration in range(config.iterations):
        rollout_rows: list[tuple[RolloutStep, float]] = []
        iteration_episodes: list[EpisodeResult] = []
        for board in boards:
            group = [
                _rollout_episode(model, tokenizer, model_cfg, board, config, rng, torch, runtime_device)
                for _ in range(config.group_size)
            ]
            iteration_episodes.extend(group)
            rollout_rows.extend(_group_training_rows(group))

        iteration_stat = {
            "iteration": iteration + 1,
            "episodes": len(iteration_episodes),
            "solved": sum(episode.solved for episode in iteration_episodes),
            "mean_reward": round(fmean(episode.reward for episode in iteration_episodes), 6),
            "mean_steps": round(fmean(episode.steps for episode in iteration_episodes), 3),
            "policy_decisions": sum(len(episode.decisions) for episode in iteration_episodes),
            "trainable_decisions": len(rollout_rows),
            "outcomes": {reason: sum(episode.reason == reason for episode in iteration_episodes) for reason in sorted({episode.reason for episode in iteration_episodes})},
        }
        stats.append(iteration_stat)
        print(json.dumps(iteration_stat, sort_keys=True), flush=True)
        if not rollout_rows:
            continue

        rng.shuffle(rollout_rows)
        model.train()
        if config.freeze_encoder:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(rollout_rows), config.batch_size):
            rows = rollout_rows[offset : offset + config.batch_size]
            items = []
            for decision, _advantage in rows:
                board = Board.from_ascii(decision.board)
                legal = tuple(Action(value) for value in decision.legal_actions)
                items.append(_encode_policy_item(board, legal, tokenizer, model_cfg, decision.action_index))
            batch = _collate(items, tokenizer.pad_token_id, torch)
            mask = batch["marker_mask"].to(runtime_device)
            old_log_probabilities = torch.tensor([decision.old_log_probability for decision, _ in rows], device=runtime_device, dtype=torch.float32)
            advantages = torch.tensor([advantage for _, advantage in rows], device=runtime_device, dtype=torch.float32)
            precision = torch.autocast("cuda", dtype=torch.float16) if runtime_device.type == "cuda" else nullcontext()
            with precision:
                logits, _ = model(
                    batch["input_ids"].to(runtime_device),
                    batch["attention_mask"].to(runtime_device),
                    batch["marker_pos"].to(runtime_device),
                    mask,
                    batch["qtype"].to(runtime_device),
                    detach_encoder=config.freeze_encoder,
                )
                scaled = _temperature_scaled(logits.float(), mask, model_cfg, torch)
                log_probabilities = torch.log_softmax(scaled.masked_fill(~mask, -1e4), dim=-1)
                chosen_log_probabilities = log_probabilities.gather(1, batch["label"].to(runtime_device).unsqueeze(1)).squeeze(1)
                log_ratio = chosen_log_probabilities - old_log_probabilities
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                probabilities = torch.softmax(scaled.masked_fill(~mask, -1e4), dim=-1)
                entropy = -(probabilities * log_probabilities * mask).sum(-1).mean()

                expert_sample = [expert_items[rng.randrange(len(expert_items))] for _ in rows]
                expert_batch = _collate(expert_sample, tokenizer.pad_token_id, torch)
                expert_mask = expert_batch["marker_mask"].to(runtime_device)
                expert_logits, _ = model(
                    expert_batch["input_ids"].to(runtime_device),
                    expert_batch["attention_mask"].to(runtime_device),
                    expert_batch["marker_pos"].to(runtime_device),
                    expert_mask,
                    expert_batch["qtype"].to(runtime_device),
                    detach_encoder=config.freeze_encoder,
                )
                expert_scaled = _temperature_scaled(expert_logits.float(), expert_mask, model_cfg, torch)
                expert_probabilities = torch.softmax(expert_scaled.masked_fill(~expert_mask, -1e4), dim=-1)
                expert_loss = -proper_reward(
                    expert_probabilities,
                    expert_batch["target"].to(runtime_device),
                    expert_batch["qtype"].to(runtime_device),
                    expert_mask,
                ).mean()
                loss = (policy_loss + config.expert_weight * expert_loss - config.entropy_weight * entropy) / config.gradient_accumulation

            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            update_index = offset // config.batch_size + 1
            final_batch = offset + config.batch_size >= len(rollout_rows)
            if update_index % config.gradient_accumulation == 0 or final_batch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        model.eval()

    destination = Path(output_dir)
    model_cfg["environment_grpo"] = True
    save_checkpoint(model, tokenizer, model_cfg, destination, stats)
    return destination
