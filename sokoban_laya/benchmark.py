"""Reproducible model-only Sokoban benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
import time
from typing import Any

from .checkpoint import load_checkpoint
from .device import resolve_device
from .grpo_training import _temperature_scaled, initial_boards
from .macro_grpo_training import _encode_macro_item, _macro_state_key
from .macros import apply_push_macro, legal_push_macros
from .training import _collate
from .trajectories import read_jsonl


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    device: str = "mps"
    max_levels: int | None = None
    max_pushes: int = 32
    batch_size: int = 32


def _summarize(states: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = {
        reason: sum(state["reason"] == reason for state in states)
        for reason in sorted({state["reason"] for state in states})
    }
    solved = outcomes.get("solved", 0)
    return {
        "levels": len(states),
        "solved": solved,
        "solve_rate": round(solved / max(1, len(states)), 4),
        "mean_pushes": round(fmean(state["pushes"] for state in states), 3),
        "outcomes": outcomes,
    }


def evaluate_macro_dataset(
    model_id_or_path: str | Path,
    dataset_path: str | Path,
    config: BenchmarkConfig = BenchmarkConfig(),
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    device = resolve_device(config.device)
    model, tokenizer, model_cfg = load_checkpoint(model_id_or_path, device)
    boards = initial_boards(read_jsonl(dataset_path), config.max_levels)
    difficulties: list[str | None] = [None] * len(boards)
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        dataset_name = Path(dataset_path).stem
        records = manifest["splits"][dataset_name]["records"][: len(boards)]
        difficulties = [record["difficulty"] for record in records]
    states = [
        {"board": board, "seen": set(), "pushes": 0, "reason": None, "difficulty": difficulty}
        for board, difficulty in zip(boards, difficulties)
    ]
    model.eval()

    for _round in range(config.max_pushes):
        selections = {}
        requests = []
        for index, state in enumerate(states):
            if state["reason"] is not None:
                continue
            board = state["board"]
            if board.is_solved():
                state["reason"] = "solved"
                continue
            key = _macro_state_key(board)
            if key in state["seen"]:
                state["reason"] = "repeat"
                continue
            state["seen"].add(key)
            macros = legal_push_macros(board)
            if not macros:
                state["reason"] = "no_reachable_push"
            elif len(macros) == 1:
                selections[index] = macros[0]
            else:
                requests.append((index, macros, _encode_macro_item(board, macros, tokenizer, model_cfg, 0)))

        for offset in range(0, len(requests), config.batch_size):
            chunk = requests[offset : offset + config.batch_size]
            batch = _collate([request[2] for request in chunk], tokenizer.pad_token_id, torch)
            mask = batch["marker_mask"].to(device)
            with torch.no_grad():
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    mask,
                    batch["qtype"].to(device),
                    detach_encoder=True,
                )
                scaled = _temperature_scaled(logits.float(), mask, model_cfg, torch).cpu()
            for row, (state_index, macros, _item) in enumerate(chunk):
                selections[state_index] = macros[int(scaled[row, : len(macros)].argmax())]

        if not selections:
            break
        for index, macro in selections.items():
            state = states[index]
            board, _primitives = apply_push_macro(state["board"], macro)
            state["board"] = board
            state["pushes"] += 1
            if board.is_solved():
                state["reason"] = "solved"
            elif board.has_static_deadlock():
                state["reason"] = "static_deadlock"

    for state in states:
        if state["reason"] is None:
            state["reason"] = "push_limit"
    result = {
        "schema_version": 1,
        "model": str(model_id_or_path),
        "dataset": str(dataset_path),
        "config": asdict(config),
        **_summarize(states),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    named_difficulties = sorted({difficulty for difficulty in difficulties if difficulty is not None})
    if named_difficulties:
        result["by_difficulty"] = {
            difficulty: _summarize([state for state in states if state["difficulty"] == difficulty])
            for difficulty in named_difficulties
        }
    return result


def save_benchmark(result: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
