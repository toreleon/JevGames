"""Reinforcement Learning for Calibrated Decisions (RLCD)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from jevgames.contracts import DecisionModelAdapter, TrainingStrategy
from jevgames.registry import register_strategy


@dataclass(frozen=True, slots=True)
class RLCDConfig:
    epochs: int = 4
    batch_size: int = 8
    gradient_accumulation: int = 2
    group_size: int = 4
    sigma_start: float = 0.5
    sigma_end: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    logging_steps: int = 25

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("rlcd.epochs must be non-negative")
        if self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("RLCD batch sizes must be positive")
        if self.group_size < 2:
            raise ValueError("rlcd.group_size must be at least two")
        if self.sigma_start <= 0.0 or self.sigma_end <= 0.0:
            raise ValueError("RLCD exploration sigma must be positive")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("rlcd.mixed_precision must be one of: no, fp16, bf16")


class _ItemDataset:
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def group_relative_advantages(rewards: Sequence[Sequence[float]]) -> list[list[float]]:
    """Normalize each evidence item's sampled rewards across its group."""

    if not rewards:
        return []
    width = len(rewards[0])
    if any(len(row) != width for row in rewards):
        raise ValueError("reward groups must have equal width")
    columns = list(zip(*rewards))
    normalized_columns: list[list[float]] = []
    for values in columns:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        deviation = variance**0.5
        if deviation < 1e-6:
            normalized_columns.append([0.0] * len(values))
        else:
            normalized_columns.append([(value - mean) / (deviation + 1e-6) for value in values])
    return [list(row) for row in zip(*normalized_columns)]


class RLCDStrategy(TrainingStrategy):
    name = "rlcd"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = RLCDConfig(**config)

    def train(
        self,
        adapter: DecisionModelAdapter,
        items: Sequence[dict[str, Any]],
        output_dir: str | Path,
        seed: int,
    ) -> list[dict[str, Any]]:
        if not items:
            raise ValueError("RLCD requires calibration evidence with at least two options")
        if self.config.epochs == 0:
            return []

        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader

        started = time.perf_counter()
        random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        data = DataLoader(
            _ItemDataset(items),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=adapter.collate,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        )
        parameters = adapter.trainable_parameters()
        if not parameters:
            raise ValueError("model adapter exposed no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation,
        )
        adapter.model, optimizer, data = accelerator.prepare(adapter.model, optimizer, data)
        adapter.device = accelerator.device
        parameters = [parameter for parameter in adapter.model.parameters() if parameter.requires_grad]
        stats: list[dict[str, Any]] = []
        global_step = 0

        for epoch in range(self.config.epochs):
            epoch_started = time.perf_counter()
            fraction = epoch / max(1, self.config.epochs - 1)
            sigma = self.config.sigma_start + fraction * (self.config.sigma_end - self.config.sigma_start)
            loss_total = 0.0
            reward_total = 0.0
            advantage_total = 0.0
            example_total = 0
            adapter.model.train()

            for batch in data:
                batch = _to_device(batch, accelerator.device)
                with accelerator.accumulate(adapter.model):
                    logits = adapter.logits(adapter.model, batch)
                    mask = adapter.action_mask(batch).bool()
                    option_count = mask.sum(-1, keepdim=True).clamp(min=1).to(logits.dtype)
                    noise = torch.randn(
                        (self.config.group_size,) + tuple(logits.shape),
                        device=logits.device,
                        dtype=logits.dtype,
                    ) * sigma
                    expanded_mask = mask.unsqueeze(0)
                    noise = noise * expanded_mask
                    noise = noise - noise.sum(-1, keepdim=True) / option_count.unsqueeze(0)
                    noise = noise * expanded_mask
                    sampled_logits = logits.detach().unsqueeze(0) + noise
                    probabilities = torch.softmax(
                        sampled_logits.masked_fill(~expanded_mask, -1e4),
                        dim=-1,
                    )
                    with torch.no_grad():
                        reward = adapter.calibration_reward(probabilities, batch)
                        centered = reward - reward.mean(dim=0, keepdim=True)
                        deviation = reward.std(dim=0, correction=0, keepdim=True)
                        advantage = torch.where(
                            deviation > 1e-6,
                            centered / (deviation + 1e-6),
                            torch.zeros_like(centered),
                        )
                    log_probability = -(
                        ((sampled_logits - logits.unsqueeze(0)) ** 2) * expanded_mask
                    ).sum(-1) / (2.0 * sigma**2)
                    loss = -(advantage * log_probability).mean()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, self.config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_examples = int(logits.shape[0])
                example_total += batch_examples
                loss_total += float(loss.detach()) * batch_examples
                reward_total += float(reward.mean()) * batch_examples
                advantage_total += float(advantage.abs().mean()) * batch_examples
                global_step += 1
                if accelerator.is_main_process and (
                    global_step == 1 or global_step % self.config.logging_steps == 0
                ):
                    print(json.dumps({
                        "phase": "rlcd",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "sigma": round(sigma, 6),
                        "loss": round(float(loss.detach()), 6),
                        "mean_proper_score": round(float(reward.mean()), 6),
                    }, sort_keys=True), flush=True)

            stat = {
                "phase": "rlcd",
                "framework": "accelerate",
                "epoch": epoch + 1,
                "examples": example_total,
                "group_size": self.config.group_size,
                "sigma": round(sigma, 6),
                "mean_loss": round(loss_total / max(1, example_total), 6),
                "mean_proper_score": round(reward_total / max(1, example_total), 6),
                "mean_absolute_advantage": round(advantage_total / max(1, example_total), 6),
                "seconds": round(time.perf_counter() - epoch_started, 3),
            }
            stats.append(stat)
            if accelerator.is_main_process:
                print(json.dumps(stat, sort_keys=True), flush=True)

        adapter.model = accelerator.unwrap_model(adapter.model)
        adapter.device = accelerator.device
        stats.append({
            "phase": "rlcd_summary",
            "epochs": self.config.epochs,
            "examples": len(items),
            "sampled_distributions_per_example": self.config.group_size,
            "seconds": round(time.perf_counter() - started, 3),
        })
        return stats


@register_strategy("rlcd")
def create_rlcd_strategy(config: dict[str, Any]) -> RLCDStrategy:
    return RLCDStrategy(config)
