"""Generic offline warm-up, online GRPO, and evaluation engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import fmean, pstdev
import time
from typing import Any, Mapping, Sequence

from .config import BenchmarkConfig, OnlineConfig, WarmupConfig
from .contracts import (
    DecisionModelAdapter,
    DecisionOption,
    DecisionTask,
    EpisodeResult,
    ExpertDecision,
    PolicyDecisionRecord,
)


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _expert_items(adapter: DecisionModelAdapter, examples: Sequence[ExpertDecision]) -> list[dict[str, Any]]:
    items = []
    for example in examples:
        if len(example.options) < 2:
            continue
        keys = tuple(option.key for option in example.options)
        if example.target_key not in keys:
            raise ValueError(f"expert target {example.target_key!r} is absent from options")
        items.append(
            adapter.encode(
                example.observation,
                example.instruction,
                example.options,
                keys.index(example.target_key),
            )
        )
    if not items:
        raise ValueError("dataset has no expert decisions with at least two options")
    return items


class _Dataset:
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def supervised_warmup(
    adapter: DecisionModelAdapter,
    items: Sequence[dict[str, Any]],
    config: WarmupConfig,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    class AdapterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = adapter.supervised_loss(model, inputs)
            outputs = {"loss": loss, "logits": adapter.logits(model, inputs)}
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(Path(output_dir) / "warmup"),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=float(config.epochs),
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation,
        max_grad_norm=config.max_grad_norm,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=seed,
        data_seed=seed,
        disable_tqdm=True,
    )
    trainer = AdapterTrainer(
        model=adapter.model,
        args=arguments,
        train_dataset=_Dataset(items),
        data_collator=adapter.collate,
    )
    result = trainer.train()
    adapter.model.eval()
    return {
        "phase": "supervised_warmup",
        "framework": "transformers.Trainer",
        "examples": len(items),
        "epochs": config.epochs,
        "train_loss": round(float(result.training_loss), 6),
        "seconds": round(float(result.metrics.get("train_runtime", 0.0)), 3),
        "samples_per_second": round(float(result.metrics.get("train_samples_per_second", 0.0)), 3),
    }


@dataclass(slots=True)
class _LiveEpisode:
    state: Any
    seen: set[Any]
    reward: float = 0.0
    environment_steps: int = 0
    primitive_steps: int = 0
    reason: str | None = None
    records: list[PolicyDecisionRecord] | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []


def _rollout_groups(
    adapter: DecisionModelAdapter,
    task: DecisionTask,
    states: Sequence[Any],
    config: OnlineConfig,
) -> list[list[EpisodeResult]]:
    import torch
    from torchrl.modules import MaskedCategorical

    live = [_LiveEpisode(state, set()) for state in states for _ in range(config.group_size)]
    model = adapter.model
    model.eval()
    for _decision_round in range(config.max_decisions):
        selections: dict[int, DecisionOption] = {}
        requests: list[tuple[int, tuple[DecisionOption, ...], dict[str, Any]]] = []
        for index, episode in enumerate(live):
            if episode.reason is not None:
                continue
            if task.is_solved(episode.state):
                episode.reason = "solved"
                continue
            key = task.state_key(episode.state)
            if key in episode.seen:
                episode.reward += task.repeated_state_reward()
                episode.reason = "repeat"
                continue
            episode.seen.add(key)
            options = task.options(episode.state)
            if not options:
                episode.reward += task.no_action_reward()
                episode.reason = "no_action"
            elif len(options) == 1:
                selections[index] = options[0]
            else:
                requests.append((
                    index,
                    options,
                    adapter.encode(task.observation(episode.state), task.instruction, options),
                ))

        for offset in range(0, len(requests), config.rollout_inference_batch_size):
            chunk = requests[offset : offset + config.rollout_inference_batch_size]
            batch = _to_device(adapter.collate([request[2] for request in chunk]), adapter.device)
            with torch.no_grad():
                logits = adapter.logits(model, batch)
                mask = adapter.action_mask(batch)
                distribution = MaskedCategorical(logits=logits, mask=mask)
                actions = distribution.sample()
                log_probabilities = distribution.log_prob(actions)
            for row, (episode_index, options, _item) in enumerate(chunk):
                action_index = int(actions[row].cpu())
                selections[episode_index] = options[action_index]
                assert live[episode_index].records is not None
                live[episode_index].records.append(
                    PolicyDecisionRecord(
                        task.serialize_state(live[episode_index].state),
                        task.instruction,
                        options,
                        action_index,
                        float(log_probabilities[row].cpu()),
                    )
                )

        if not selections:
            break
        for index, option in selections.items():
            episode = live[index]
            outcome = task.transition(episode.state, option.key)
            episode.state = outcome.state
            episode.reward += outcome.reward
            episode.environment_steps += 1
            episode.primitive_steps += outcome.primitive_steps
            if outcome.terminated:
                episode.reason = outcome.reason or ("solved" if outcome.solved else "terminated")

    results = []
    for episode in live:
        if episode.reason is None:
            episode.reward += task.limit_reward()
            episode.reason = "decision_limit"
        results.append(
            EpisodeResult(
                reward=round(episode.reward, 6),
                solved=task.is_solved(episode.state),
                decisions=len(episode.records or ()),
                environment_steps=episode.environment_steps,
                primitive_steps=episode.primitive_steps,
                reason=episode.reason,
                policy_records=tuple(episode.records or ()),
            )
        )
    return [results[index : index + config.group_size] for index in range(0, len(results), config.group_size)]


def _advantaged_records(group: Sequence[EpisodeResult]):
    rewards = [episode.reward for episode in group]
    deviation = pstdev(rewards)
    if deviation < 1e-6:
        return []
    mean = fmean(rewards)
    rows = []
    for episode in group:
        advantage = (episode.reward - mean) / (deviation + 1e-6)
        rows.extend((record, advantage) for record in episode.policy_records)
    return rows


def online_grpo(
    adapter: DecisionModelAdapter,
    task: DecisionTask,
    expert_items: Sequence[dict[str, Any]],
    initial_states: Sequence[Any],
    config: OnlineConfig,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from accelerate import Accelerator
    from torchrl.modules import MaskedCategorical

    rng = random.Random(seed)
    torch.manual_seed(seed)
    parameters = adapter.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=config.gradient_accumulation,
    )
    adapter.model, optimizer = accelerator.prepare(adapter.model, optimizer)
    adapter.device = accelerator.device
    parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
    stats = []

    for iteration in range(config.iterations):
        started = time.perf_counter()
        episodes = []
        rows = []
        for offset in range(0, len(initial_states), config.rollout_instance_batch_size):
            groups = _rollout_groups(
                adapter,
                task,
                initial_states[offset : offset + config.rollout_instance_batch_size],
                config,
            )
            for group in groups:
                episodes.extend(group)
                rows.extend(_advantaged_records(group))
        rng.shuffle(rows)
        adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(rows), config.batch_size):
            sample = rows[offset : offset + config.batch_size]
            items = []
            for record, _advantage in sample:
                state = task.deserialize_state(record.serialized_state)
                items.append(adapter.encode(
                    task.observation(state),
                    record.instruction,
                    record.options,
                    record.action_index,
                ))
            with accelerator.accumulate(adapter.model):
                batch = _to_device(adapter.collate(items), accelerator.device)
                logits = adapter.logits(adapter.model, batch)
                distribution = MaskedCategorical(logits=logits, mask=adapter.action_mask(batch))
                chosen = distribution.log_prob(adapter.labels(batch))
                old_logp = torch.tensor([record.old_log_probability for record, _ in sample], device=accelerator.device)
                advantage = torch.tensor([value for _, value in sample], device=accelerator.device)
                ratio = torch.exp(chosen - old_logp)
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage,
                ).mean()
                entropy = distribution.entropy().mean()
                replay = [expert_items[rng.randrange(len(expert_items))] for _ in sample]
                expert_batch = _to_device(adapter.collate(replay), accelerator.device)
                expert_loss = adapter.supervised_loss(adapter.model, expert_batch)
                loss = policy_loss + config.expert_weight * expert_loss - config.entropy_weight * entropy
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        adapter.model.eval()
        outcomes = {
            reason: sum(episode.reason == reason for episode in episodes)
            for reason in sorted({episode.reason for episode in episodes})
        }
        stat = {
            "phase": "online_grpo",
            "framework": "accelerate+torchrl",
            "iteration": iteration + 1,
            "episodes": len(episodes),
            "solved": sum(episode.solved for episode in episodes),
            "mean_reward": round(fmean(episode.reward for episode in episodes), 6),
            "trainable_decisions": len(rows),
            "outcomes": outcomes,
            "seconds": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(stat, sort_keys=True), flush=True)
        stats.append(stat)

    adapter.model = accelerator.unwrap_model(adapter.model)
    adapter.device = accelerator.device
    return stats


def evaluate(
    adapter: DecisionModelAdapter,
    task: DecisionTask,
    states: Sequence[Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    import torch
    from torchrl.modules import MaskedCategorical

    selected_states = list(states[: config.max_instances] if config.max_instances else states)
    live = [_LiveEpisode(state, set()) for state in selected_states]
    adapter.model.eval()
    for _round in range(config.max_decisions):
        selections = {}
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
                requests.append((index, options, adapter.encode(task.observation(episode.state), task.instruction, options)))
        for offset in range(0, len(requests), config.batch_size):
            chunk = requests[offset : offset + config.batch_size]
            batch = _to_device(adapter.collate([request[2] for request in chunk]), adapter.device)
            with torch.no_grad():
                distribution = MaskedCategorical(
                    logits=adapter.logits(adapter.model, batch),
                    mask=adapter.action_mask(batch),
                )
                actions = distribution.mode
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
        "solve_rate": round(solved / max(1, len(live)), 4),
        "mean_environment_steps": round(fmean(episode.environment_steps for episode in live), 3),
        "mean_primitive_steps": round(fmean(episode.primitive_steps for episode in live), 3),
        "outcomes": outcomes,
    }
