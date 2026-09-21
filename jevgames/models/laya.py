"""Laya adapter for the generic decision-training contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jevgames.contracts import DecisionKind, DecisionModelAdapter, DecisionQuestion
from jevgames.registry import register_model
from sokoban_laya.checkpoint import load_checkpoint, save_checkpoint
from sokoban_laya.device import resolve_device
from sokoban_laya.training import _collate


def _temperature_scaled(logits: Any, mask: Any, qtypes: Any, model_cfg: dict[str, Any]):
    import torch
    from laya.common import QTYPES, temp_bucket

    defaults = model_cfg.get("temperature", [1.0, 1.0, 1.0])
    by_options = model_cfg.get("temperature_by_options", {})
    scales = []
    for qtype, count in zip(qtypes.tolist(), mask.sum(-1).tolist()):
        value = by_options.get(temp_bucket(int(qtype), int(count)), defaults[int(qtype)])
        scales.append(max(1e-3, float(value)))
    scale = torch.tensor(scales, device=logits.device, dtype=logits.dtype).unsqueeze(-1)
    return logits / scale


class LayaAdapter(DecisionModelAdapter):
    name = "laya"

    def __init__(self, config: dict[str, Any]) -> None:
        self.source = str(config.get("source", "convaiinnovations/laya"))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.model, self.tokenizer, self.model_cfg = load_checkpoint(self.source, self.device)
        self.model.eval()
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
        question: DecisionQuestion,
        target_probabilities: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        from laya.common import QTYPES, build_sequence

        if len(question.options) < 2:
            raise ValueError("Laya requires at least two options")
        if question.kind is DecisionKind.CHOICE:
            criteria: Any = {option.key: option.description for option in question.options}
        elif question.kind is DecisionKind.SCORE:
            criteria = [option.description for option in question.options]
        else:
            criteria = {option.key: option.description for option in question.options}
        native_question = {
            "t": question.kind.value,
            "ins": question.instruction,
            "crit": criteria,
        }
        ids, markers = build_sequence(
            self.tokenizer,
            dict(observation),
            native_question,
            self.model_cfg["max_len"],
            self.model_cfg["head_max_len"],
        )
        if len(markers) != len(question.options):
            raise ValueError("decision options exceeded Laya's head token budget")
        if target_probabilities is None:
            target = [1.0 / len(question.options)] * len(question.options)
        else:
            target = [float(value) for value in target_probabilities]
            if len(target) != len(question.options):
                raise ValueError("target probabilities must match the Laya options")
            if abs(sum(target) - 1.0) > 1e-6:
                raise ValueError("target probabilities must sum to one")
        target_index = max(range(len(target)), key=target.__getitem__)
        return {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[question.kind.value],
            "target": target,
            "label": target_index,
        }

    def collate(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return _collate(list(items), self.tokenizer.pad_token_id, torch)

    def _raw_logits(self, active_model: Any, batch: Mapping[str, Any]):
        logits, _ = active_model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
            detach_encoder=self._freeze_backbone,
        )
        return logits.float()

    def logits(self, active_model: Any, batch: Mapping[str, Any]):
        return _temperature_scaled(
            self._raw_logits(active_model, batch),
            batch["marker_mask"],
            batch["qtype"],
            self.model_cfg,
        )

    def training_logits(self, active_model: Any, batch: Mapping[str, Any]):
        return self._raw_logits(active_model, batch)

    def action_mask(self, batch: Mapping[str, Any]):
        return batch["marker_mask"]

    def target_probabilities(self, batch: Mapping[str, Any]):
        return batch["target"]

    def ordinal_mask(self, batch: Mapping[str, Any]):
        from laya.common import QTYPES

        return batch["qtype"] == QTYPES["score"]

    def freeze_backbone(self) -> None:
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)

    def fit_calibration(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int,
    ) -> Mapping[str, Any]:
        import torch
        from laya.common import QTYPES, temp_bucket

        def fit_temperature(samples: list[tuple[Any, Any]]) -> float | None:
            if len(samples) < 10:
                return None
            width = max(len(logits) for logits, _target in samples)
            logits = torch.full((len(samples), width), -1e4, dtype=torch.float32)
            targets = torch.zeros((len(samples), width), dtype=torch.float32)
            for row, (sample_logits, sample_target) in enumerate(samples):
                logits[row, : len(sample_logits)] = sample_logits
                targets[row, : len(sample_target)] = sample_target
            log_temperature = torch.zeros(1, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

            def closure():
                optimizer.zero_grad()
                loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), dim=-1)).sum(-1).mean()
                loss.backward()
                return loss

            optimizer.step(closure)
            return float(log_temperature.exp().clamp(0.1, 10.0).item())

        by_type: dict[int, list[tuple[Any, Any]]] = {value: [] for value in QTYPES.values()}
        by_bucket: dict[str, list[tuple[Any, Any]]] = {}
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(items), batch_size):
                batch = self.collate(items[offset : offset + batch_size])
                batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                logits = self._raw_logits(self.model, batch).cpu()
                targets = batch["target"].cpu()
                mask = batch["marker_mask"].cpu()
                qtypes = batch["qtype"].cpu()
                for row in range(int(logits.shape[0])):
                    count = int(mask[row].sum().item())
                    qtype = int(qtypes[row].item())
                    sample = (logits[row, :count], targets[row, :count])
                    by_type[qtype].append(sample)
                    by_bucket.setdefault(temp_bucket(qtype, count), []).append(sample)

        temperatures = list(self.model_cfg.get("temperature", [1.0, 1.0, 1.0]))
        while len(temperatures) < len(QTYPES):
            temperatures.append(1.0)
        fitted_types = {}
        for qtype, samples in by_type.items():
            fitted = fit_temperature(samples)
            if fitted is not None:
                temperatures[qtype] = fitted
                fitted_types[next(name for name, value in QTYPES.items() if value == qtype)] = round(fitted, 6)

        fitted_type_names = set(fitted_types)
        temperatures_by_options = {
            bucket: value
            for bucket, value in self.model_cfg.get("temperature_by_options", {}).items()
            if bucket.split(":", 1)[0] not in fitted_type_names
        }
        fitted_buckets = {}
        for bucket, samples in by_bucket.items():
            fitted = fit_temperature(samples)
            if fitted is not None:
                temperatures_by_options[bucket] = fitted
                fitted_buckets[bucket] = round(fitted, 6)
        self.model_cfg["temperature"] = temperatures
        self.model_cfg["temperature_by_options"] = temperatures_by_options
        return {
            "method": "temperature_scaling",
            "fitted_types": fitted_types,
            "fitted_option_buckets": fitted_buckets,
        }

    def save(self, model: Any, destination: str | Path, metadata: Mapping[str, Any]) -> Path:
        config = dict(self.model_cfg)
        config["jev_games_framework"] = True
        return save_checkpoint(model, self.tokenizer, config, destination, [dict(metadata)])


@register_model("laya")
def create_laya_adapter(config: dict[str, Any]) -> LayaAdapter:
    return LayaAdapter(config)
