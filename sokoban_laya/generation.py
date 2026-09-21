"""Procedural solvable Sokoban generation by reverse play.

Levels start solved.  The generator repeatedly performs legal reverse pulls,
recording the exact inverse primitive actions.  Reversing those actions gives
a verified forward solution without running an expensive solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Mapping

from .core import Action, Board, Position
from .macros import DIRECTIONS, reachable_walk_paths
from .solver import replay


_INVERSE_MOVE: Mapping[Action, Action] = {
    Action.MOVE_UP: Action.MOVE_DOWN,
    Action.MOVE_DOWN: Action.MOVE_UP,
    Action.MOVE_LEFT: Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
}


@dataclass(frozen=True, slots=True)
class Difficulty:
    name: str
    width: int
    height: int
    boxes: int
    reverse_pushes_min: int
    reverse_pushes_max: int
    wall_probability: float


PILOT_DIFFICULTIES = (
    Difficulty("easy_2box", 7, 7, 2, 4, 8, 0.05),
    Difficulty("medium_3box", 8, 8, 3, 8, 14, 0.08),
    Difficulty("hard_4box", 10, 10, 4, 12, 20, 0.10),
)


@dataclass(frozen=True, slots=True)
class GeneratedLevel:
    board: Board
    solution: tuple[Action, ...]
    difficulty: str
    seed: int
    reverse_pushes: int

    @property
    def layout_id(self) -> str:
        payload = repr((self.board.width, self.board.height, sorted(self.board.walls), sorted(self.board.goals))).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    @property
    def level_id(self) -> str:
        payload = repr((self.layout_id, sorted(self.board.boxes), self.board.player)).encode()
        return hashlib.sha256(payload).hexdigest()[:20]


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


def _largest_floor_component(width: int, height: int, walls: set[Position]) -> set[Position]:
    remaining = {
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
        if (row, col) not in walls
    }
    components: list[set[Position]] = []
    while remaining:
        start = next(iter(remaining))
        component = {start}
        frontier = [start]
        remaining.remove(start)
        while frontier:
            row, col = frontier.pop()
            for candidate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if candidate in remaining:
                    remaining.remove(candidate)
                    component.add(candidate)
                    frontier.append(candidate)
        components.append(component)
    return max(components, key=len, default=set())


def _topology(difficulty: Difficulty, rng: random.Random) -> tuple[frozenset[Position], tuple[Position, ...]]:
    width, height = difficulty.width, difficulty.height
    walls = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if row in (0, height - 1) or col in (0, width - 1)
    }
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if rng.random() < difficulty.wall_probability:
                walls.add((row, col))
    floor = _largest_floor_component(width, height, walls)
    if len(floor) < difficulty.boxes * 5 + 3:
        raise ValueError("generated topology has insufficient connected floor")
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if (row, col) not in floor:
                walls.add((row, col))
    return frozenset(walls), tuple(sorted(floor))


def _reverse_pull_candidates(board: Board):
    paths = reachable_walk_paths(board)
    candidates = []
    for player_spot, walking_path in paths.items():
        for direction in DIRECTIONS:
            box = _add(player_spot, direction.delta)
            destination_player = _add(player_spot, (-direction.delta[0], -direction.delta[1]))
            if box not in board.boxes:
                continue
            if not board.is_floor(destination_player) or destination_player in board.boxes:
                continue
            boxes = set(board.boxes)
            boxes.remove(box)
            boxes.add(player_spot)
            result = Board(
                board.width,
                board.height,
                board.walls,
                board.goals,
                frozenset(boxes),
                destination_player,
            )
            candidates.append((walking_path, direction, result))
    return candidates


def generate_level(difficulty: Difficulty, seed: int, max_attempts: int = 100) -> GeneratedLevel:
    rng = random.Random(seed)
    for _attempt in range(max_attempts):
        try:
            walls, floor = _topology(difficulty, rng)
        except ValueError:
            continue
        goals = frozenset(rng.sample(floor, difficulty.boxes))
        player_choices = [position for position in floor if position not in goals]
        if not player_choices:
            continue
        solved = Board(
            difficulty.width,
            difficulty.height,
            walls,
            goals,
            goals,
            rng.choice(player_choices),
        )
        board = solved
        inverse_actions: list[Action] = []
        seen = {(board.boxes, board.player)}
        target_pulls = rng.randint(difficulty.reverse_pushes_min, difficulty.reverse_pushes_max)
        pulls = 0
        while pulls < target_pulls:
            candidates = []
            for walking_path, direction, result in _reverse_pull_candidates(board):
                key = (result.boxes, result.player)
                if key not in seen:
                    candidates.append((walking_path, direction, result))
            if not candidates:
                break
            walking_path, direction, result = rng.choice(candidates)
            current = board
            for move in walking_path:
                transition = current.apply(move)
                if not transition.moved or transition.pushed:
                    raise RuntimeError("reverse walking path unexpectedly moved a box")
                current = transition.board
                inverse_actions.append(_INVERSE_MOVE[move])
            # The precomputed result includes the walk and one reverse pull.
            if current.player != _add(result.player, direction.delta):
                raise RuntimeError("reverse pull candidate and walking path disagree")
            board = result
            inverse_actions.append(direction.push)
            seen.add((board.boxes, board.player))
            pulls += 1
        if pulls < difficulty.reverse_pushes_min or board.is_solved():
            continue
        if len(board.boxes - board.goals) < max(1, difficulty.boxes // 2):
            continue
        solution = tuple(reversed(inverse_actions))
        if not replay(board, solution).is_solved():
            raise RuntimeError("reverse-generated solution failed replay verification")
        return GeneratedLevel(board, solution, difficulty.name, seed, pulls)
    raise RuntimeError(f"could not generate {difficulty.name!r} level from seed {seed}")


def difficulty_schedule(count: int) -> Iterable[Difficulty]:
    """40% easy, 35% medium, 25% hard, deterministically interleaved."""

    for index in range(count):
        bucket = index % 20
        if bucket < 8:
            yield PILOT_DIFFICULTIES[0]
        elif bucket < 15:
            yield PILOT_DIFFICULTIES[1]
        else:
            yield PILOT_DIFFICULTIES[2]
