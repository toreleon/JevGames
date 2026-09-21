"""Laya adapter for the generic decision-training contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jevgames.contracts import DecisionModelAdapter, DecisionOption
from jevgames.registry import register_model
from sokoban_laya.checkpoint import load_checkpoint, save_checkpoint
from sokoban_laya.device import resolve_device
from sokoban_laya.grpo_training import _temperature_scaled
from sokoban_laya.training import _collate


class LayaAdapter(DecisionModelAdapter):
    name = "laya"

    def __init__(self, config: dict[str, Any]) -> None:
        self.source = str(config.get("source", "convaiinnovations/laya"))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.model, self.tokenizer, self.model_cfg = load_checkpoint(self.source, self.device)
        self._freeze_backbone = bool(config.get("freeze_backbone", True))
        if self._freeze_backbone:
            self.freeze_backbone()

    def encode(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        options: Sequence[DecisionOption],
        target_index: int = 0,
    ) -> dict[str, Any]:
        from laya.common import QTYPES, build_sequence

        if len(options) < 2:
            raise ValueError("Laya requires at least two options")
        question = {
            "t": "choice",
            "ins": instruction,
            "crit": {option.key: option.description for option in options},
        }
        ids, markers = build_sequence(
            self.tokenizer,
            dict(observation),
            question,
            self.model_cfg["max_len"],
            self.model_cfg["head_max_len"],
        )
        if len(markers) != len(options):
            raise ValueError("decision options exceeded Laya's head token budget")
        target = [0.0] * len(options)
        target[target_index] = 1.0
        return {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES["choice"],
            "target": target,
            "label": target_index,
        }

    def collate(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return _collate(list(items), self.tokenizer.pad_token_id, torch)

    def logits(self, active_model: Any, batch: Mapping[str, Any]):
        logits, _ = active_model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
            detach_encoder=self._freeze_backbone,
        )
        import torch

        return _temperature_scaled(logits.float(), batch["marker_mask"], self.model_cfg, torch)

    def action_mask(self, batch: Mapping[str, Any]):
        return batch["marker_mask"]

    def labels(self, batch: Mapping[str, Any]):
        return batch["label"]

    def supervised_loss(self, active_model: Any, batch: Mapping[str, Any]):
        import torch
        from laya.common import proper_reward

        logits = self.logits(active_model, batch)
        mask = self.action_mask(batch)
        probabilities = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
        proper_loss = -proper_reward(
            probabilities,
            batch["target"],
            batch["qtype"],
            mask,
        ).mean()
        classification_loss = torch.nn.functional.cross_entropy(logits, self.labels(batch))
        return classification_loss + 0.25 * proper_loss

    def freeze_backbone(self) -> None:
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)

    def save(self, model: Any, destination: str | Path, metadata: Mapping[str, Any]) -> Path:
        config = dict(self.model_cfg)
        config["jev_games_framework"] = True
        return save_checkpoint(model, self.tokenizer, config, destination, [dict(metadata)])


@register_model("laya")
def create_laya_adapter(config: dict[str, Any]) -> LayaAdapter:
    return LayaAdapter(config)
