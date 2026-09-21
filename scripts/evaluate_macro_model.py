"""Greedy model-only evaluation using reachable push macros and no search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laya

from sokoban_laya.core import Board
from sokoban_laya.device import resolve_device
from sokoban_laya.laya_policy import BOARD_LEGEND
from sokoban_laya.macros import apply_push_macro, legal_push_macros, macro_question, reachable_walk_paths


def load_board(path: Path) -> Board:
    text = path.read_text(encoding="utf-8")
    return Board.from_xsb(text) if path.suffix.lower() == ".xsb" else Board.from_ascii(text)


def state_key(board: Board):
    return board.boxes, frozenset(reachable_walk_paths(board))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("level")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-pushes", type=int, default=100)
    args = parser.parse_args()

    board = load_board(Path(args.level))
    agent = laya.load(args.model, device=str(resolve_device(args.device)))
    seen = set()
    primitive_steps = model_decisions = forced_pushes = 0
    trace: list[str] = []
    reason = "push_limit"
    for _ in range(args.max_pushes):
        if board.is_solved():
            reason = "solved"
            break
        key = state_key(board)
        if key in seen:
            reason = "repeat"
            break
        if board.has_static_deadlock():
            reason = "static_deadlock"
            break
        seen.add(key)
        macros = legal_push_macros(board)
        if not macros:
            reason = "no_reachable_push"
            break
        if len(macros) == 1:
            selected = macros[0]
            forced_pushes += 1
        else:
            result = agent.predict(
                {"board": board.render(), "legend": BOARD_LEGEND},
                {"push": macro_question(macros)},
            )
            label = result["answers"]["push"]["choice"]
            selected = next(macro for macro in macros if macro.label == label)
            model_decisions += 1
        board, primitives = apply_push_macro(board, selected)
        primitive_steps += len(primitives)
        trace.append(selected.label)
    else:
        reason = "push_limit"
    if board.is_solved():
        reason = "solved"
    print(json.dumps({
        "model": args.model,
        "level": args.level,
        "result": reason,
        "solved": board.is_solved(),
        "pushes": len(trace),
        "primitive_steps": primitive_steps,
        "model_decisions": model_decisions,
        "forced_pushes": forced_pushes,
        "trace": trace,
    }))


if __name__ == "__main__":
    main()
