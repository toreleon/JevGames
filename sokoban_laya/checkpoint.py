"""Laya-compatible checkpoint loading and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_checkpoint(model_id_or_path: str | Path, device: Any):
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_dir = Path(model_id_or_path)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(str(model_id_or_path)))
    _fix_tokenizer_config(str(model_dir))
    with (model_dir / "rl_agent_config.json").open(encoding="utf-8") as handle:
        model_cfg = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = build_model(model_cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    model.to(device)
    return model, tokenizer, model_cfg


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any],
    destination: str | Path,
    stats: list[dict[str, Any]],
) -> Path:
    from safetensors.torch import save_file

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    state = {name: value.half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(state, output / "model.safetensors")
    model.encoder.config.save_pretrained(output / "encoder")
    tokenizer.save_pretrained(output / "tokenizer")
    config = dict(model_cfg)
    config["fine_tuned"] = True
    with (output / "rl_agent_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output / "training_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return output
