"""Small-to-medium-board A* Sokoban solver used for expert demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import permutations
from typing import Iterable

from .core import Action, Board


@dataclass(frozen=True, slots=True)
class SolveResult:
    actions: tuple[Action, ...]
    expanded: int

    @property
    def pushes(self) -> int:
        return sum(action.value.startswith("push_") for action in self.actions)


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _assignment_lower_bound(board: Board) -> int:
    """Admissible box-to-goal matching lower bound for the A* heuristic."""

    boxes = tuple(sorted(board.boxes))
    goals = tuple(sorted(board.goals))
    if not boxes:
        return 0
    return min(sum(_manhattan(box, goal) for box, goal in zip(boxes, ordered_goals)) for ordered_goals in permutations(goals))


def solve(initial: Board, max_expansions: int = 200_000) -> SolveResult | None:
    """Find a shortest-action solution with sound static-deadlock pruning.

    This is intentionally a transparent data-generation solver rather than a
    world-class Sokoban engine.  It is well suited to curriculum levels up to
    a few boxes; difficult large instances may hit ``max_expansions``.
    """

    if initial.is_solved():
        return SolveResult((), 0)
    # Walls and goals never change.  Calculate this board-topology property
    # once rather than repeating a reverse search for every A* successor.
    dead_squares = initial.static_dead_squares()

    def has_static_deadlock(board: Board) -> bool:
        return any(box not in board.goals and box in dead_squares for box in board.boxes)

    if has_static_deadlock(initial):
        return None

    counter = 0
    frontier: list[tuple[int, int, int, Board, tuple[Action, ...]]] = []
    heapq.heappush(frontier, (_assignment_lower_bound(initial), 0, counter, initial, ()))
    best_cost: dict[Board, int] = {initial: 0}
    expanded = 0

    while frontier and expanded < max_expansions:
        _f_score, cost, _tie, board, path = heapq.heappop(frontier)
        if cost != best_cost.get(board):
            continue
        expanded += 1
        for action in board.legal_actions():
            child = board.apply(action).board
            child_cost = cost + 1
            if child_cost >= best_cost.get(child, float("inf")):
                continue
            if has_static_deadlock(child):
                continue
            child_path = path + (action,)
            if child.is_solved():
                return SolveResult(child_path, expanded)
            best_cost[child] = child_cost
            counter += 1
            estimate = child_cost + _assignment_lower_bound(child)
            heapq.heappush(frontier, (estimate, child_cost, counter, child, child_path))
    return None


def replay(initial: Board, actions: Iterable[Action]) -> Board:
    """Apply a sequence and raise if it is not a valid trajectory."""

    board = initial
    for step, action in enumerate(actions):
        transition = board.apply(action)
        if not transition.moved:
            raise ValueError(f"invalid action {action.value!r} at trajectory step {step}")
        board = transition.board
    return board
