"""Framework integrations for scalable Laya training.

Transformers Trainer owns offline expert warm-up.  Accelerate owns online-RL
optimization mechanics.  Domain-specific Sokoban rollout and reward code stays
outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class WarmupSettings:
    epochs: int = 6
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 11
    logging_steps: int = 25


class ItemDataset:
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def run_macro_warmup(
    model: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
    expert_items: Sequence[dict[str, Any]],
    output_dir: str | Path,
    settings: WarmupSettings,
) -> dict[str, Any]:
    """Run calibrated macro imitation through Hugging Face Trainer."""

    import torch
    from laya.common import proper_reward
    from transformers import Trainer, TrainingArguments

    from .grpo_training import _temperature_scaled
    from .training import _collate

    class ProperScoringTrainer(Trainer):
        def compute_loss(self, active_model, inputs, return_outputs=False, num_items_in_batch=None):
            marker_mask = inputs["marker_mask"]
            logits, _ = active_model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["marker_pos"],
                marker_mask,
                inputs["qtype"],
                detach_encoder=True,
            )
            scaled = _temperature_scaled(logits.float(), marker_mask, model_cfg, torch)
            probabilities = torch.softmax(scaled.masked_fill(~marker_mask, -1e4), dim=-1)
            calibrated_loss = -proper_reward(
                probabilities,
                inputs["target"],
                inputs["qtype"],
                marker_mask,
            ).mean()
            classification_loss = torch.nn.functional.cross_entropy(scaled, inputs["label"])
            loss = classification_loss + 0.25 * calibrated_loss
            outputs = {"logits": scaled, "loss": loss}
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(Path(output_dir) / "hf_warmup"),
        per_device_train_batch_size=settings.batch_size,
        num_train_epochs=float(settings.epochs),
        learning_rate=settings.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=settings.weight_decay,
        gradient_accumulation_steps=settings.gradient_accumulation,
        max_grad_norm=1.0,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        logging_strategy="steps",
        logging_steps=settings.logging_steps,
        logging_first_step=True,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=settings.seed,
        data_seed=settings.seed,
        disable_tqdm=True,
    )
    trainer = ProperScoringTrainer(
        model=model,
        args=arguments,
        train_dataset=ItemDataset(expert_items),
        data_collator=lambda rows: _collate(rows, tokenizer.pad_token_id, torch),
    )
    result = trainer.train()
    model.eval()
    return {
        "phase": "hf_trainer_macro_warmup",
        "epochs": settings.epochs,
        "examples": len(expert_items),
        "train_loss": round(float(result.training_loss), 6),
        "seconds": round(float(result.metrics.get("train_runtime", 0.0)), 3),
        "samples_per_second": round(float(result.metrics.get("train_samples_per_second", 0.0)), 3),
        "steps_per_second": round(float(result.metrics.get("train_steps_per_second", 0.0)), 3),
    }


class AccelerateOptimizer:
    """Thin owner of standard distributed optimization mechanics."""

    def __init__(self, model: Any, optimizer: Any, gradient_accumulation: int) -> None:
        from accelerate import Accelerator

        self.accelerator = Accelerator(
            mixed_precision="no",
            gradient_accumulation_steps=gradient_accumulation,
        )
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)

    def accumulation_context(self):
        return self.accelerator.accumulate(self.model)

    def backward_and_step(self, loss: Any, parameters: Sequence[Any]) -> None:
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def unwrap_model(self) -> Any:
        return self.accelerator.unwrap_model(self.model)
