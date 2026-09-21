"""Push-macro environment GRPO for Sokoban.

The policy chooses only strategic box pushes.  A deterministic BFS controller
executes the walking path required for each push, drastically shortening the
credit-assignment horizon compared with primitive move actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
import time
from typing import Any, Sequence

from .core import Action, Board
from .checkpoint import load_checkpoint, save_checkpoint
from .device import resolve_device
from .grpo_training import _temperature_scaled, initial_boards
from .laya_policy import BOARD_LEGEND
from .macros import PushMacro, apply_push_macro, legal_push_macros, macro_question, reachable_walk_paths
from .training import _collate
from .training_framework import AccelerateOptimizer, WarmupSettings, run_macro_warmup
from .trajectories import TrainingExample, read_jsonl


@dataclass(frozen=True, slots=True)
class MacroGRPOConfig:
    expert_warmup_epochs: int = 6
    iterations: int = 2
    group_size: int = 4
    max_pushes: int = 32
    max_levels: int = 4
    batch_size: int = 4
    rollout_level_batch_size: int = 16
    rollout_inference_batch_size: int = 32
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    expert_weight: float = 0.40
    gradient_accumulation: int = 2
    freeze_encoder: bool = True
    seed: int = 11
    solve_reward: float = 50.0
    goal_reward: float = 4.0
    off_goal_penalty: float = 4.0
    push_penalty: float = 0.02
    walk_penalty: float = 0.002
    deadlock_penalty: float = 15.0
    repeat_penalty: float = 4.0
    limit_penalty: float = 1.0


@dataclass(frozen=True, slots=True)
class MacroExpertExample:
    board: str
    expert_label: str
    episode: str
    push_step: int


@dataclass(frozen=True, slots=True)
class MacroRolloutStep:
    board: str
    action_index: int
    old_log_probability: float


@dataclass(frozen=True, slots=True)
class MacroEpisodeResult:
    reward: float
    solved: bool
    pushes: int
    primitive_steps: int
    reason: str
    decisions: tuple[MacroRolloutStep, ...]


@dataclass(slots=True)
class _LiveMacroEpisode:
    board: Board
    seen: set[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]
    reward: float = 0.0
    pushes: int = 0
    primitive_steps: int = 0
    reason: str | None = None
    decisions: list[MacroRolloutStep] | None = None

    def __post_init__(self) -> None:
        if self.decisions is None:
            self.decisions = []


def compress_expert_trajectories(examples: Sequence[TrainingExample]) -> list[MacroExpertExample]:
    """Compress primitive solver paths into one labelled state per box push."""

    grouped: dict[str, list[TrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.episode, []).append(example)
    compressed: list[MacroExpertExample] = []
    for episode, records in grouped.items():
        records.sort(key=lambda record: record.step)
        board = Board.from_ascii(records[0].board)
        macro_start = board
        push_step = 0
        for record in records:
            action = Action(record.expert_action)
            transition = board.apply(action)
            if not transition.moved:
                raise ValueError(f"invalid expert trajectory in episode {episode!r} at step {record.step}")
            board = transition.board
            if not action.value.startswith("push_"):
                continue
            candidates = []
            for macro in legal_push_macros(macro_start):
                candidate_board, _ = apply_push_macro(macro_start, macro)
                if candidate_board == board:
                    candidates.append(macro)
            if len(candidates) != 1:
                raise ValueError(
                    f"could not uniquely map primitive push to a macro in episode {episode!r}, step {record.step}"
                )
            compressed.append(MacroExpertExample(macro_start.render(), candidates[0].label, episode, push_step))
            macro_start = board
            push_step += 1
        if not board.is_solved():
            raise ValueError(f"expert episode {episode!r} does not end solved")
    return compressed


def _internal_macro_question(macros: tuple[PushMacro, ...]) -> dict[str, Any]:
    external = macro_question(macros)
    return {"t": "choice", "ins": external["instructions"], "crit": external["criteria"]}


def _encode_macro_item(board: Board, macros: tuple[PushMacro, ...], tokenizer: Any, model_cfg: dict[str, Any], label: int) -> dict[str, Any]:
    from laya.common import QTYPES, build_sequence

    question = _internal_macro_question(macros)
    ids, markers = build_sequence(
        tokenizer,
        {"board": board.render(), "legend": BOARD_LEGEND},
        question,
        model_cfg["max_len"],
        model_cfg["head_max_len"],
    )
    if len(markers) != len(macros):
        raise ValueError("push-macro options exceeded Laya's head token budget")
    target = [0.0] * len(macros)
    target[label] = 1.0
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target, "label": label}


def _build_expert_items(examples: Sequence[MacroExpertExample], tokenizer: Any, model_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for example in examples:
        board = Board.from_ascii(example.board)
        macros = legal_push_macros(board)
        if len(macros) < 2:
            continue
        labels = tuple(macro.label for macro in macros)
        if example.expert_label not in labels:
            raise ValueError(f"expert macro {example.expert_label!r} is unavailable")
        items.append(_encode_macro_item(board, macros, tokenizer, model_cfg, labels.index(example.expert_label)))
    if not items:
        raise ValueError("macro expert dataset has no states with at least two legal pushes")
    return items


def _macro_state_key(board: Board) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Treat all player positions in the same reachable region as one state."""

    return board.boxes, frozenset(reachable_walk_paths(board))


def _rollout_macro_episode(model: Any, tokenizer: Any, model_cfg: dict[str, Any], initial: Board, config: MacroGRPOConfig, rng: random.Random, torch: Any, device: Any) -> MacroEpisodeResult:
    board = initial
    seen: set[tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]] = set()
    decisions: list[MacroRolloutStep] = []
    reward = 0.0
    primitive_steps = 0
    reason = "push_limit"
    model.eval()

    for push_index in range(config.max_pushes):
        if board.is_solved():
            reason = "solved"
            break
        state_key = _macro_state_key(board)
        if state_key in seen:
            reward -= config.repeat_penalty
            reason = "repeat"
            break
        seen.add(state_key)
        macros = legal_push_macros(board)
        if not macros:
            reward -= config.deadlock_penalty
            reason = "no_reachable_push"
            break

        if len(macros) == 1:
            selected = macros[0]
        else:
            item = _encode_macro_item(board, macros, tokenizer, model_cfg, 0)
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
                probabilities = torch.softmax(scaled.masked_fill(~mask, -1e4), dim=-1)[0, : len(macros)].cpu().tolist()
            action_index = rng.choices(range(len(macros)), weights=probabilities, k=1)[0]
            selected = macros[action_index]
            decisions.append(MacroRolloutStep(board.render(), action_index, math.log(max(1e-8, probabilities[action_index]))))

        before_goals = len(board.boxes & board.goals)
        board, primitives = apply_push_macro(board, selected)
        primitive_steps += len(primitives)
        reward -= config.push_penalty + config.walk_penalty * max(0, len(primitives) - 1)
        after_goals = len(board.boxes & board.goals)
        delta = after_goals - before_goals
        reward += max(0, delta) * config.goal_reward
        reward -= max(0, -delta) * config.off_goal_penalty
        if board.is_solved():
            reward += config.solve_reward
            reason = "solved"
            push_index += 1
            break
        if board.has_static_deadlock():
            reward -= config.deadlock_penalty
            reason = "static_deadlock"
            push_index += 1
            break
    else:
        reward -= config.limit_penalty
        push_index = config.max_pushes

    return MacroEpisodeResult(
        reward=round(reward, 6),
        solved=board.is_solved(),
        pushes=int(push_index),
        primitive_steps=primitive_steps,
        reason=reason,
        decisions=tuple(decisions),
    )


def _rollout_macro_groups_batched(
    model: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
    boards: Sequence[Board],
    config: MacroGRPOConfig,
    rng: random.Random,
    torch: Any,
    device: Any,
) -> list[list[MacroEpisodeResult]]:
    """Roll out many level groups while batching every Laya forward pass."""

    live = [
        _LiveMacroEpisode(board=board, seen=set())
        for board in boards
        for _ in range(config.group_size)
    ]
    model.eval()
    for _push_round in range(config.max_pushes):
        selections: dict[int, PushMacro] = {}
        requests: list[tuple[int, tuple[PushMacro, ...], dict[str, Any]]] = []
        for index, episode in enumerate(live):
            if episode.reason is not None:
                continue
            if episode.board.is_solved():
                episode.reason = "solved"
                continue
            key = _macro_state_key(episode.board)
            if key in episode.seen:
                episode.reward -= config.repeat_penalty
                episode.reason = "repeat"
                continue
            episode.seen.add(key)
            macros = legal_push_macros(episode.board)
            if not macros:
                episode.reward -= config.deadlock_penalty
                episode.reason = "no_reachable_push"
                continue
            if len(macros) == 1:
                selections[index] = macros[0]
            else:
                requests.append((index, macros, _encode_macro_item(episode.board, macros, tokenizer, model_cfg, 0)))

        for offset in range(0, len(requests), config.rollout_inference_batch_size):
            from torchrl.modules import MaskedCategorical

            chunk = requests[offset : offset + config.rollout_inference_batch_size]
            batch = _collate([request[2] for request in chunk], tokenizer.pad_token_id, torch)
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
                distribution = MaskedCategorical(logits=scaled, mask=mask)
                sampled_actions = distribution.sample()
                sampled_log_probabilities = distribution.log_prob(sampled_actions)
            for row, (episode_index, macros, _item) in enumerate(chunk):
                action_index = int(sampled_actions[row].cpu())
                selections[episode_index] = macros[action_index]
                assert live[episode_index].decisions is not None
                live[episode_index].decisions.append(
                    MacroRolloutStep(
                        live[episode_index].board.render(),
                        action_index,
                        float(sampled_log_probabilities[row].cpu()),
                    )
                )

        if not selections:
            break
        for index, selected in selections.items():
            episode = live[index]
            before_goals = len(episode.board.boxes & episode.board.goals)
            episode.board, primitives = apply_push_macro(episode.board, selected)
            episode.pushes += 1
            episode.primitive_steps += len(primitives)
            episode.reward -= config.push_penalty + config.walk_penalty * max(0, len(primitives) - 1)
            after_goals = len(episode.board.boxes & episode.board.goals)
            delta = after_goals - before_goals
            episode.reward += max(0, delta) * config.goal_reward
            episode.reward -= max(0, -delta) * config.off_goal_penalty
            if episode.board.is_solved():
                episode.reward += config.solve_reward
                episode.reason = "solved"
            elif episode.board.has_static_deadlock():
                episode.reward -= config.deadlock_penalty
                episode.reason = "static_deadlock"

    results = []
    for episode in live:
        if episode.reason is None:
            episode.reward -= config.limit_penalty
            episode.reason = "push_limit"
        results.append(
            MacroEpisodeResult(
                reward=round(episode.reward, 6),
                solved=episode.board.is_solved(),
                pushes=episode.pushes,
                primitive_steps=episode.primitive_steps,
                reason=episode.reason,
                decisions=tuple(episode.decisions or ()),
            )
        )
    return [
        results[index : index + config.group_size]
        for index in range(0, len(results), config.group_size)
    ]


def _group_rows(episodes: Sequence[MacroEpisodeResult]) -> list[tuple[MacroRolloutStep, float]]:
    rewards = [episode.reward for episode in episodes]
    mean = fmean(rewards)
    deviation = pstdev(rewards)
    if deviation < 1e-6:
        return []
    rows = []
    for episode in episodes:
        advantage = (episode.reward - mean) / (deviation + 1e-6)
        rows.extend((decision, advantage) for decision in episode.decisions)
    return rows


def train_macro_grpo(
    model_id_or_path: str | Path,
    low_level_expert_dataset: str | Path,
    output_dir: str | Path,
    config: MacroGRPOConfig = MacroGRPOConfig(),
    device: str = "auto",
) -> Path:
    try:
        import torch
        from laya.common import proper_reward
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install model dependencies with: uv sync --extra laya") from exc

    training_started = time.perf_counter()
    runtime_device = resolve_device(device)
    load_started = time.perf_counter()
    model, tokenizer, model_cfg = load_checkpoint(model_id_or_path, runtime_device)
    low_level_examples = read_jsonl(low_level_expert_dataset)
    macro_examples = compress_expert_trajectories(low_level_examples)
    expert_items = _build_expert_items(macro_examples, tokenizer, model_cfg)
    boards = initial_boards(low_level_examples, config.max_levels)
    setup_seconds = time.perf_counter() - load_started
    rng = random.Random(config.seed)

    if config.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    stats: list[dict[str, Any]] = [{
        "phase": "setup",
        "seconds": round(setup_seconds, 3),
        "primitive_expert_steps": len(low_level_examples),
        "macro_expert_steps": len(macro_examples),
        "trainable_macro_steps": len(expert_items),
        "rollout_levels": len(boards),
    }]

    print(json.dumps({"primitive_expert_steps": len(low_level_examples), "macro_expert_steps": len(macro_examples)}), flush=True)
    if config.expert_warmup_epochs > 0:
        warmup_stat = run_macro_warmup(
            model,
            tokenizer,
            model_cfg,
            expert_items,
            output_dir,
            WarmupSettings(
                epochs=config.expert_warmup_epochs,
                batch_size=config.batch_size,
                gradient_accumulation=config.gradient_accumulation,
                learning_rate=config.learning_rate,
                weight_decay=0.01,
                seed=config.seed,
            ),
        )
        stats.append(warmup_stat)
        print(json.dumps(warmup_stat), flush=True)
    model.eval()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.01)
    optimization = AccelerateOptimizer(model, optimizer, config.gradient_accumulation)
    model = optimization.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    runtime_device = optimization.accelerator.device

    for iteration in range(config.iterations):
        phase_started = time.perf_counter()
        episodes: list[MacroEpisodeResult] = []
        rollout_rows: list[tuple[MacroRolloutStep, float]] = []
        for board_offset in range(0, len(boards), config.rollout_level_batch_size):
            level_batch = boards[board_offset : board_offset + config.rollout_level_batch_size]
            groups = _rollout_macro_groups_batched(
                model, tokenizer, model_cfg, level_batch, config, rng, torch, runtime_device
            )
            for group in groups:
                episodes.extend(group)
                rollout_rows.extend(_group_rows(group))
        stat = {
            "phase": "environment_grpo",
            "iteration": iteration + 1,
            "episodes": len(episodes),
            "solved": sum(episode.solved for episode in episodes),
            "mean_reward": round(fmean(episode.reward for episode in episodes), 6),
            "mean_pushes": round(fmean(episode.pushes for episode in episodes), 3),
            "mean_primitive_steps": round(fmean(episode.primitive_steps for episode in episodes), 3),
            "policy_decisions": sum(len(episode.decisions) for episode in episodes),
            "trainable_decisions": len(rollout_rows),
            "outcomes": {reason: sum(episode.reason == reason for episode in episodes) for reason in sorted({episode.reason for episode in episodes})},
        }
        stats.append(stat)
        print(json.dumps(stat, sort_keys=True), flush=True)
        if not rollout_rows:
            stat["seconds"] = round(time.perf_counter() - phase_started, 3)
            continue

        rng.shuffle(rollout_rows)
        model.train()
        if config.freeze_encoder:
            model.encoder.eval()
        for offset in range(0, len(rollout_rows), config.batch_size):
            rows = rollout_rows[offset : offset + config.batch_size]
            items = []
            for decision, _ in rows:
                board = Board.from_ascii(decision.board)
                macros = legal_push_macros(board)
                items.append(_encode_macro_item(board, macros, tokenizer, model_cfg, decision.action_index))
            batch = _collate(items, tokenizer.pad_token_id, torch)
            mask = batch["marker_mask"].to(runtime_device)
            old_log_probabilities = torch.tensor([decision.old_log_probability for decision, _ in rows], device=runtime_device, dtype=torch.float32)
            advantages = torch.tensor([advantage for _, advantage in rows], device=runtime_device, dtype=torch.float32)
            with optimization.accumulation_context():
                from torchrl.modules import MaskedCategorical

                logits, _ = model(
                    batch["input_ids"].to(runtime_device),
                    batch["attention_mask"].to(runtime_device),
                    batch["marker_pos"].to(runtime_device),
                    mask,
                    batch["qtype"].to(runtime_device),
                    detach_encoder=config.freeze_encoder,
                )
                scaled = _temperature_scaled(logits.float(), mask, model_cfg, torch)
                distribution = MaskedCategorical(logits=scaled, mask=mask)
                chosen = distribution.log_prob(batch["label"].to(runtime_device))
                ratio = torch.exp(chosen - old_log_probabilities)
                policy_loss = -torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages,
                ).mean()
                entropy = distribution.entropy().mean()

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
                loss = policy_loss + config.expert_weight * expert_loss - config.entropy_weight * entropy
                optimization.backward_and_step(loss, trainable)
        model.eval()
        stat["seconds"] = round(time.perf_counter() - phase_started, 3)

    output_cfg = dict(model_cfg)
    output_cfg["push_macro_grpo"] = True
    destination = Path(output_dir)
    save_started = time.perf_counter()
    model = optimization.unwrap_model()
    save_checkpoint(model, tokenizer, output_cfg, destination, stats)
    stats.append({
        "phase": "total",
        "save_seconds": round(time.perf_counter() - save_started, 3),
        "elapsed_seconds": round(time.perf_counter() - training_started, 3),
    })
    with (destination / "grpo_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return destination
