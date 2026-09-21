"""Transform solved Sokoban trajectories into decision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core import Action, Board
from .macros import apply_push_macro, legal_push_macros
from .trajectories import TrainingExample


@dataclass(frozen=True, slots=True)
class MacroExpertExample:
    board: str
    expert_label: str
    episode: str
    push_step: int


def compress_expert_trajectories(examples: Sequence[TrainingExample]) -> list[MacroExpertExample]:
    """Compress primitive solver paths into one observed outcome per box push."""

    grouped: dict[str, list[TrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.episode, []).append(example)
    compressed: list[MacroExpertExample] = []
    for episode, records in grouped.items():
        records.sort(key=lambda record: record.step)
        board = Board.from_ascii(records[0].board)
        macro_start = board
        push_step = 0
        for record in records:
            action = Action(record.expert_action)
            transition = board.apply(action)
            if not transition.moved:
                raise ValueError(f"invalid expert trajectory in episode {episode!r} at step {record.step}")
            board = transition.board
            if not action.value.startswith("push_"):
                continue
            candidates = []
            for macro in legal_push_macros(macro_start):
                candidate_board, _ = apply_push_macro(macro_start, macro)
                if candidate_board == board:
                    candidates.append(macro)
            if len(candidates) != 1:
                raise ValueError(
                    f"could not uniquely map primitive push to a macro in episode {episode!r}, step {record.step}"
                )
            compressed.append(MacroExpertExample(macro_start.render(), candidates[0].label, episode, push_step))
            macro_start = board
            push_step += 1
        if not board.is_solved():
            raise ValueError(f"expert episode {episode!r} does not end solved")
    return compressed
