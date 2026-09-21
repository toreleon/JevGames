"""Reachable push macros that remove low-level walking from policy decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Mapping

from .core import Action, Board, Position


@dataclass(frozen=True, slots=True)
class Direction:
    name: str
    delta: Position
    move: Action
    push: Action


DIRECTIONS: Final[tuple[Direction, ...]] = (
    Direction("up", (-1, 0), Action.MOVE_UP, Action.PUSH_UP),
    Direction("down", (1, 0), Action.MOVE_DOWN, Action.PUSH_DOWN),
    Direction("left", (0, -1), Action.MOVE_LEFT, Action.PUSH_LEFT),
    Direction("right", (0, 1), Action.MOVE_RIGHT, Action.PUSH_RIGHT),
)


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


@dataclass(frozen=True, slots=True)
class PushMacro:
    """One reachable box push plus the deterministic shortest walking path."""

    box: Position
    direction: str
    push_action: Action
    walking_actions: tuple[Action, ...]

    @property
    def label(self) -> str:
        return f"box_r{self.box[0]}_c{self.box[1]}_push_{self.direction}"

    @property
    def description(self) -> str:
        row, column = self.box[0] + 1, self.box[1] + 1
        return (
            f"Push the box currently at row {row}, column {column} one square {self.direction}; "
            "the controller walks the player to the required side first"
        )

    @property
    def low_level_actions(self) -> tuple[Action, ...]:
        return self.walking_actions + (self.push_action,)


def reachable_walk_paths(board: Board) -> Mapping[Position, tuple[Action, ...]]:
    """Shortest paths the player can walk without moving any box."""

    paths: dict[Position, tuple[Action, ...]] = {board.player: ()}
    queue: deque[Position] = deque([board.player])
    moves = tuple((direction.delta, direction.move) for direction in DIRECTIONS)
    while queue:
        current = queue.popleft()
        for delta, action in moves:
            candidate = _add(current, delta)
            if candidate in paths or not board.is_floor(candidate) or candidate in board.boxes:
                continue
            paths[candidate] = paths[current] + (action,)
            queue.append(candidate)
    return paths


def legal_push_macros(board: Board) -> tuple[PushMacro, ...]:
    """Enumerate every box push reachable without moving another box."""

    paths = reachable_walk_paths(board)
    macros: list[PushMacro] = []
    for box in sorted(board.boxes):
        for direction in DIRECTIONS:
            destination = _add(box, direction.delta)
            stand = _add(box, (-direction.delta[0], -direction.delta[1]))
            if not board.is_floor(destination) or destination in board.boxes or stand not in paths:
                continue
            macros.append(PushMacro(box, direction.name, direction.push, paths[stand]))
    return tuple(macros)


def apply_push_macro(board: Board, macro: PushMacro) -> tuple[Board, tuple[Action, ...]]:
    """Execute a macro and return the resulting board and primitive actions."""

    available = {candidate.label: candidate for candidate in legal_push_macros(board)}
    if macro.label not in available:
        raise ValueError(f"push macro {macro.label!r} is not legal in this state")
    selected = available[macro.label]
    current = board
    for action in selected.low_level_actions:
        transition = current.apply(action)
        if not transition.moved:
            raise RuntimeError(f"macro path contains invalid primitive action {action.value!r}")
        current = transition.board
    return current, selected.low_level_actions


def macro_question(macros: tuple[PushMacro, ...]) -> dict[str, object]:
    if len(macros) < 2:
        raise ValueError("Laya requires at least two push options; execute forced macros directly")
    return {
        "type": "choice",
        "instructions": (
            "Choose the reachable box push most likely to solve the Sokoban level. "
            "Avoid irreversible deadlocks and preserve access to every remaining box."
        ),
        "criteria": {macro.label: macro.description for macro in macros},
    }
