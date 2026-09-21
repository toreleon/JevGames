"""Convert solver solutions into JSONL records for Laya fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .core import Action, Board
from .solver import SolveResult, replay, solve


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One policy decision at one Sokoban state.

    `legal_actions` is part of the question, rather than an inference-only
    mask.  The model therefore never needs to learn that a wall is illegal;
    it learns to choose between meaningful, executable actions.
    """

    board: str
    legal_actions: tuple[str, ...]
    expert_action: str
    episode: str
    step: int


def solution_trajectory(initial: Board, result: SolveResult, episode: str) -> list[TrainingExample]:
    """Create one supervised example per pre-action state in a solution."""

    examples: list[TrainingExample] = []
    board = initial
    for step, action in enumerate(result.actions):
        legal = board.legal_actions()
        if action not in legal:
            raise ValueError(f"solver emitted invalid action {action.value!r} at step {step}")
        examples.append(
            TrainingExample(
                board=board.render(),
                legal_actions=tuple(candidate.value for candidate in legal),
                expert_action=action.value,
                episode=episode,
                step=step,
            )
        )
        board = board.apply(action).board
    if not board.is_solved():
        raise ValueError("only solved trajectories may be used as expert demonstrations")
    return examples


def solve_to_trajectory(initial: Board, episode: str, max_expansions: int = 200_000) -> list[TrainingExample]:
    """Solve a level and return its expert trajectory, or raise cleanly."""

    result = solve(initial, max_expansions=max_expansions)
    if result is None:
        raise RuntimeError(f"no solution found for {episode!r} within the search budget")
    return solution_trajectory(initial, result, episode)


def write_jsonl(examples: Iterable[TrainingExample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                examples.append(
                    TrainingExample(
                        board=str(row["board"]),
                        legal_actions=tuple(str(action) for action in row["legal_actions"]),
                        expert_action=str(row["expert_action"]),
                        episode=str(row.get("episode", "unknown")),
                        step=int(row.get("step", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid training record at line {line_number}") from exc
    if not examples:
        raise ValueError("training file has no examples")
    return examples


def verify_trajectory(initial: Board, examples: Iterable[TrainingExample]) -> Board:
    """Validate that saved JSONL actions replay from a given initial board."""

    actions = tuple(Action(example.expert_action) for example in examples)
    return replay(initial, actions)
