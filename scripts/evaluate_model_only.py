"""Greedy, model-only Sokoban evaluation with no search fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_laya.core import Board
from sokoban_laya.laya_policy import LayaPolicy


def load_board(path: Path) -> Board:
    text = path.read_text(encoding="utf-8")
    return Board.from_xsb(text) if path.suffix.lower() == ".xsb" else Board.from_ascii(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("level")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    board = load_board(Path(args.level))
    policy = LayaPolicy.from_hub(args.model, subfolder=None, device=args.device)
    seen: set[Board] = set()
    pushes = model_decisions = forced_moves = 0
    reason = "step_limit"
    trace: list[str] = []
    for step in range(args.max_steps):
        if board.is_solved():
            reason = "solved"
            break
        if board in seen:
            reason = "repeat"
            break
        if board.has_static_deadlock():
            reason = "static_deadlock"
            break
        seen.add(board)
        legal = board.legal_actions()
        if not legal:
            reason = "no_legal_action"
            break
        decision = policy.choose(board)
        if len(legal) == 1:
            forced_moves += 1
        else:
            model_decisions += 1
        transition = board.apply(decision.action)
        pushes += int(transition.pushed)
        trace.append(decision.action.value)
        board = transition.board
    else:
        step = args.max_steps
    if board.is_solved():
        reason = "solved"
    print(json.dumps({
        "model": args.model,
        "level": args.level,
        "result": reason,
        "solved": board.is_solved(),
        "steps": len(trace),
        "pushes": pushes,
        "model_decisions": model_decisions,
        "forced_moves": forced_moves,
        "trace": trace,
    }))


if __name__ == "__main__":
    main()
