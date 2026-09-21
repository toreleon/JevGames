"""Laya adapter for the generic decision-training contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jevgames.contracts import DecisionModelAdapter, DecisionOption
from jevgames.registry import register_model
from sokoban_laya.checkpoint import load_checkpoint, save_checkpoint
from sokoban_laya.device import resolve_device
from sokoban_laya.training import _collate


def _temperature_scaled(logits: Any, mask: Any, model_cfg: dict[str, Any]):
    import torch
    from laya.common import QTYPES, temp_bucket

    defaults = model_cfg.get("temperature", [1.0, 1.0, 1.0])
    by_options = model_cfg.get("temperature_by_options", {})
    scales = []
    for count in mask.sum(-1).tolist():
        value = by_options.get(temp_bucket(QTYPES["choice"], int(count)), defaults[QTYPES["choice"]])
        scales.append(max(1e-3, float(value)))
    scale = torch.tensor(scales, device=logits.device, dtype=logits.dtype).unsqueeze(-1)
    return logits / scale


class LayaAdapter(DecisionModelAdapter):
    name = "laya"

    def __init__(self, config: dict[str, Any]) -> None:
        self.source = str(config.get("source", "convaiinnovations/laya"))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.model, self.tokenizer, self.model_cfg = load_checkpoint(self.source, self.device)
        self._freeze_backbone = bool(config.get("freeze_backbone", True))
        if self._freeze_backbone:
            self.freeze_backbone()
        elif bool(config.get("gradient_checkpointing", True)):
            self.model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.head_checkpointing = True

    def encode(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        options: Sequence[DecisionOption],
        target_probabilities: Sequence[float] | None = None,
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
        if target_probabilities is None:
            target = [1.0 / len(options)] * len(options)
        else:
            target = [float(value) for value in target_probabilities]
            if len(target) != len(options):
                raise ValueError("target probabilities must match the Laya options")
            if abs(sum(target) - 1.0) > 1e-6:
                raise ValueError("target probabilities must sum to one")
        target_index = max(range(len(target)), key=target.__getitem__)
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
        return _temperature_scaled(logits.float(), batch["marker_mask"], self.model_cfg)

    def action_mask(self, batch: Mapping[str, Any]):
        return batch["marker_mask"]

    def target_probabilities(self, batch: Mapping[str, Any]):
        return batch["target"]

    def calibration_reward(self, probabilities: Any, batch: Mapping[str, Any]):
        from laya.common import proper_reward

        return proper_reward(
            probabilities,
            batch["target"],
            batch["qtype"],
            batch["marker_mask"],
        )

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
