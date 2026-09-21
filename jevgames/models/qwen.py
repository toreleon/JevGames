"""Causal Qwen backbone adapted to dynamic typed decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jevgames.contracts import DecisionKind, DecisionModelAdapter, DecisionQuestion
from jevgames.device import resolve_torch_device
from jevgames.registry import register_model


class QwenDecisionModel:
    """Factory namespace used to avoid importing torch at module discovery."""

    @staticmethod
    def build(backbone: Any):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self, active_backbone: Any) -> None:
                super().__init__()
                self.backbone = active_backbone
                hidden_size = int(active_backbone.config.hidden_size)
                head_size = max(64, hidden_size // 2)
                self.scorer = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, head_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(head_size, 1),
                )

            def forward(self, input_ids: Any, attention_mask: Any):
                hidden = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                final_position = attention_mask.sum(-1).clamp(min=1) - 1
                row = torch.arange(hidden.shape[0], device=hidden.device)
                pooled = hidden[row, final_position]
                return self.scorer(pooled).squeeze(-1).float()

        return _Model(backbone)


class QwenDecisionAdapter(DecisionModelAdapter):
    name = "qwen_decision"

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.source = str(config.get("source", "Qwen/Qwen3-0.6B-Base"))
        torch.manual_seed(int(config.get("seed", 42)))
        self.device = resolve_torch_device(str(config.get("device", "auto")))
        self.max_length = int(config.get("max_length", 512))
        self._freeze_backbone = bool(config.get("freeze_backbone", True))
        source_path = Path(self.source)
        native_config = source_path / "qwen_decision_config.json"
        if native_config.exists():
            saved = json.loads(native_config.read_text(encoding="utf-8"))
            self.max_length = int(saved.get("max_length", self.max_length))
            self.temperatures = list(saved.get("temperatures", [saved.get("temperature", 1.0)] * 3))
            self.tokenizer = AutoTokenizer.from_pretrained(source_path / "tokenizer")
            backbone = AutoModel.from_pretrained(source_path / "backbone", dtype=torch.float32)
            self.model = QwenDecisionModel.build(backbone)
            head_state = load_file(source_path / "decision_head.safetensors")
            self.model.scorer.load_state_dict(head_state, strict=True)
        else:
            self.temperatures = list(config.get("temperatures", [float(config.get("temperature", 1.0))] * 3))
            self.tokenizer = AutoTokenizer.from_pretrained(self.source)
            backbone = AutoModel.from_pretrained(self.source, dtype=torch.float32)
            self.model = QwenDecisionModel.build(backbone)
        while len(self.temperatures) < len(DecisionKind):
            self.temperatures.append(1.0)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device).eval()
        if self._freeze_backbone:
            self.freeze_backbone()
        elif bool(config.get("gradient_checkpointing", True)):
            self.model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @staticmethod
    def _render_state(observation: Mapping[str, Any]) -> str:
        return json.dumps(dict(observation), ensure_ascii=False, separators=(",", ":"), default=str)

    def encode(
        self,
        observation: Mapping[str, Any],
        question: DecisionQuestion,
        target_probabilities: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        if len(question.options) < 2:
            raise ValueError("Qwen decisions require at least two options")
        candidates = "\n".join(
            f"- {option.key}: {option.description}"
            for option in question.options
        )
        prefix = (
            f"State:\n{self._render_state(observation)}\n\n"
            f"Decision type: {question.kind.value}\n"
            f"Question: {question.instruction}\n"
            f"Available options:\n{candidates}\n\n"
        )
        candidate_ids = []
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            for option in question.options:
                text = prefix + f"Candidate to score: {option.key}: {option.description}\nDecision score:"
                candidate_ids.append(self.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"])
        finally:
            self.tokenizer.truncation_side = original_side
        if target_probabilities is None:
            target = [1.0 / len(question.options)] * len(question.options)
        else:
            target = [float(value) for value in target_probabilities]
            if len(target) != len(question.options) or abs(sum(target) - 1.0) > 1e-6:
                raise ValueError("Qwen target probabilities must match options and sum to one")
        return {
            "candidate_ids": candidate_ids,
            "target": target,
            "ordinal": question.kind is DecisionKind.SCORE,
            "kind": tuple(DecisionKind).index(question.kind),
        }

    def collate(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        flat_ids = []
        rows = []
        columns = []
        option_count = max(len(item["candidate_ids"]) for item in items)
        for row, item in enumerate(items):
            for column, ids in enumerate(item["candidate_ids"]):
                rows.append(row)
                columns.append(column)
                flat_ids.append(ids)
        sequence_length = max(len(ids) for ids in flat_ids)
        input_ids = torch.full(
            (len(flat_ids), sequence_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, ids in enumerate(flat_ids):
            input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, : len(ids)] = 1
        option_mask = torch.zeros((len(items), option_count), dtype=torch.bool)
        target = torch.zeros((len(items), option_count), dtype=torch.float32)
        for row, item in enumerate(items):
            count = len(item["candidate_ids"])
            option_mask[row, :count] = True
            target[row, :count] = torch.tensor(item["target"], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_rows": torch.tensor(rows, dtype=torch.long),
            "candidate_columns": torch.tensor(columns, dtype=torch.long),
            "option_mask": option_mask,
            "target": target,
            "ordinal": torch.tensor([bool(item["ordinal"]) for item in items]),
            "kind": torch.tensor([int(item["kind"]) for item in items], dtype=torch.long),
            "batch_size": len(items),
        }

    def training_logits(self, active_model: Any, batch: Mapping[str, Any]):
        scores = active_model(batch["input_ids"], batch["attention_mask"])
        logits = scores.new_full((int(batch["batch_size"]), int(batch["option_mask"].shape[1])), -1e4)
        return logits.index_put(
            (batch["candidate_rows"], batch["candidate_columns"]),
            scores,
        )

    def logits(self, active_model: Any, batch: Mapping[str, Any]):
        import torch

        temperatures = torch.tensor(self.temperatures, device=batch["kind"].device, dtype=torch.float32)
        row_temperature = temperatures[batch["kind"]].clamp_min(0.1).unsqueeze(-1)
        return self.training_logits(active_model, batch) / row_temperature

    def action_mask(self, batch: Mapping[str, Any]):
        return batch["option_mask"]

    def target_probabilities(self, batch: Mapping[str, Any]):
        return batch["target"]

    def ordinal_mask(self, batch: Mapping[str, Any]):
        return batch["ordinal"]

    def freeze_backbone(self) -> None:
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)

    def fit_calibration(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int,
    ) -> Mapping[str, Any]:
        import torch

        grouped: dict[int, list[tuple[Any, Any]]] = {0: [], 1: [], 2: []}
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(items), batch_size):
                batch = self.collate(items[offset : offset + batch_size])
                batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                logits = self.training_logits(self.model, batch).cpu()
                targets = batch["target"].cpu()
                mask = batch["option_mask"].cpu()
                kinds = batch["kind"].cpu()
                for row in range(int(logits.shape[0])):
                    count = int(mask[row].sum())
                    grouped[int(kinds[row])].append((logits[row, :count], targets[row, :count]))

        fitted = {}
        names = [kind.value for kind in DecisionKind]
        for kind, samples in grouped.items():
            if len(samples) < 10:
                continue
            width = max(len(row_logits) for row_logits, _row_target in samples)
            logits = torch.full((len(samples), width), -1e4)
            targets = torch.zeros((len(samples), width))
            for row, (row_logits, row_target) in enumerate(samples):
                logits[row, : len(row_logits)] = row_logits
                targets[row, : len(row_target)] = row_target
            log_temperature = torch.zeros(1, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

            def closure():
                optimizer.zero_grad()
                loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), -1)).sum(-1).mean()
                loss.backward()
                return loss

            optimizer.step(closure)
            self.temperatures[kind] = float(log_temperature.exp().clamp(0.1, 10.0).item())
            fitted[names[kind]] = round(self.temperatures[kind], 6)
        return {"method": "temperature_scaling", "fitted_types": fitted}

    def save(self, model: Any, destination: str | Path, metadata: Mapping[str, Any]) -> Path:
        from safetensors.torch import save_file

        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        model.backbone.save_pretrained(output / "backbone", safe_serialization=True)
        self.tokenizer.save_pretrained(output / "tokenizer")
        save_file(
            {name: value.detach().half().contiguous().cpu() for name, value in model.scorer.state_dict().items()},
            output / "decision_head.safetensors",
        )
        (output / "qwen_decision_config.json").write_text(json.dumps({
            "base_model": self.source,
            "max_length": self.max_length,
            "temperatures": self.temperatures,
            "training_metadata": dict(metadata),
        }, indent=2) + "\n", encoding="utf-8")
        return output


@register_model("qwen_decision")
def create_qwen_decision_adapter(config: dict[str, Any]) -> QwenDecisionAdapter:
    return QwenDecisionAdapter(config)
