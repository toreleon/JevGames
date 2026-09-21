"""Legacy notebook-compatible RLCD baseline.

The maintained scalable path is ``macro_grpo_training.py``, which delegates
training mechanics to Transformers Trainer, Accelerate, and TorchRL.  This
module remains for reproducing the originally supplied Colab behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
from typing import Any

from .core import ACTION_DESCRIPTIONS, Action
from .device import resolve_device
from .laya_policy import BOARD_LEGEND, question_for
from .trajectories import TrainingExample, read_jsonl


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    base_model: str = "convaiinnovations/laya"
    epochs: int = 4
    # Apple unified memory is generous but shared; a physical batch of one
    # plus accumulation is the dependable M4 Pro starting point.
    batch_size: int = 1
    learning_rate_encoder: float = 1.2e-5
    learning_rate_head: float = 1.2e-4
    gradient_accumulation: int = 8
    group_size: int = 4
    sigma_start: float = 0.5
    sigma_end: float = 0.2
    max_len: int = 512
    head_max_len: int = 192


def _question(example: TrainingExample) -> dict[str, Any]:
    legal = tuple(Action(value) for value in example.legal_actions)
    if not legal or Action(example.expert_action) not in legal:
        raise ValueError("each example needs an expert action contained in its legal actions")
    external = question_for(legal) if len(legal) >= 2 else None
    return {
        "t": "choice",
        "ins": external["instructions"] if external else "Forced Sokoban move.",
        "crit": external["criteria"] if external else {action.value: ACTION_DESCRIPTIONS[action] for action in legal},
    }


def _build_items(examples: list[TrainingExample], tokenizer: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from laya.common import QTYPES, build_sequence

    items: list[dict[str, Any]] = []
    for example in examples:
        question = _question(example)
        # The Laya head calculates top-2 uncertainty features, so rows with a
        # forced action cannot be represented.  They carry no policy-learning
        # signal anyway and are executed directly by LayaPolicy.choose().
        if len(question["crit"]) < 2:
            continue
        ids, markers = build_sequence(tokenizer, {"board": example.board, "legend": BOARD_LEGEND}, question, cfg["max_len"], cfg["head_max_len"])
        choices = tuple(question["crit"])
        target_index = choices.index(example.expert_action)
        if len(markers) != len(choices):
            raise ValueError("question exceeded head token budget; reduce action text or raise head_max_len")
        target = [0.0] * len(choices)
        target[target_index] = 1.0
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target, "label": target_index})
    if not items:
        raise ValueError("dataset has no states with at least two legal actions to train Laya")
    return items


def _collate(items: list[dict[str, Any]], pad_id: int, torch: Any) -> dict[str, Any]:
    count = len(items)
    sequence_length = max(len(item["ids"]) for item in items)
    option_count = max(len(item["markers"]) for item in items)
    ids = torch.full((count, sequence_length), pad_id, dtype=torch.long)
    attention = torch.zeros((count, sequence_length), dtype=torch.long)
    marker_positions = torch.zeros((count, option_count), dtype=torch.long)
    marker_mask = torch.zeros((count, option_count), dtype=torch.bool)
    target = torch.zeros((count, option_count), dtype=torch.float32)
    labels = torch.zeros(count, dtype=torch.long)
    qtypes = torch.zeros(count, dtype=torch.long)
    for index, item in enumerate(items):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        attention[index, : len(item["ids"])] = 1
        marker_positions[index, : len(item["markers"])] = torch.tensor(item["markers"])
        marker_mask[index, : len(item["markers"])] = True
        target[index, : len(item["target"])] = torch.tensor(item["target"])
        labels[index] = item["label"]
        qtypes[index] = item["qtype"]
    return {"input_ids": ids, "attention_mask": attention, "marker_pos": marker_positions, "marker_mask": marker_mask, "target": target, "label": labels, "qtype": qtypes}


def fine_tune(dataset: str | Path, output_dir: str | Path, config: FineTuneConfig = FineTuneConfig(), device: str = "auto") -> Path:
    """Fine-tune a Laya checkpoint on solver demonstrations and save it.

    This follows the notebook's RLCD policy-gradient formulation.  The model
    chooses among only state-valid actions, so its probability distribution is
    trained on the same decision space it sees during play.
    """

    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer
        from laya.agent import _fix_tokenizer_config
        from laya.common import build_model, proper_reward
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Install the optional model dependencies: pip install -e '.[laya]'") from exc

    examples = read_jsonl(dataset)
    model_dir = snapshot_download(config.base_model)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as handle:
        model_cfg = json.load(handle)
    model_cfg.update({"gradient_checkpointing": True, "max_len": config.max_len, "head_max_len": config.head_max_len, "head_layers": 2})
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    items = _build_items(examples, tokenizer, model_cfg)
    model = build_model(model_cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    runtime_device = resolve_device(device)
    if runtime_device.type in {"cuda", "mps"}:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(runtime_device).train()

    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.")]
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": config.learning_rate_encoder},
        {"params": head_parameters, "lr": config.learning_rate_head},
    ], weight_decay=0.01)
    # Laya itself uses float32 on MPS.  That is deliberate: Metal does not
    # support float64 and an M4 fine-tune is more stable in float32 than the
    # notebook's T4-specific fp16 + CUDA GradScaler path.
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime_device.type == "cuda" else None
    rng = random.Random(0)

    for epoch in range(config.epochs):
        rng.shuffle(items)
        sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * epoch / max(1, config.epochs - 1)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(items), config.batch_size):
            batch = _collate(items[offset : offset + config.batch_size], tokenizer.pad_token_id, torch)
            mask = batch["marker_mask"].to(runtime_device)
            precision_context = torch.autocast("cuda", dtype=torch.float16) if runtime_device.type == "cuda" else nullcontext()
            with precision_context:
                logits, _ = model(
                    batch["input_ids"].to(runtime_device), batch["attention_mask"].to(runtime_device),
                    batch["marker_pos"].to(runtime_device), mask, batch["qtype"].to(runtime_device),
                )
                logits = logits.float()
                option_count = mask.sum(-1, keepdim=True).float()
                noise = torch.randn((config.group_size,) + logits.shape, device=runtime_device) * sigma * mask
                noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
                sampled_logits = logits.detach().unsqueeze(0) + noise
                probabilities = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), dim=-1)
                with torch.no_grad():
                    reward = proper_reward(probabilities, batch["target"].to(runtime_device).unsqueeze(0), batch["qtype"].to(runtime_device), mask)
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std() + 1e-6)
                log_probability = -(((sampled_logits - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
                loss = -(advantage * log_probability).mean() / config.gradient_accumulation
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            if (offset // config.batch_size + 1) % config.gradient_accumulation == 0 or offset + config.batch_size >= len(items):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state = {name: value.half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(state, destination / "model.safetensors")
    model.encoder.config.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "tokenizer")
    model_cfg["fine_tuned"] = True
    with (destination / "rl_agent_config.json").open("w", encoding="utf-8") as handle:
        json.dump(model_cfg, handle, indent=2)
    return destination
