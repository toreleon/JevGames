"""Pure, deterministic Sokoban rules and a compact ASCII board encoding.

The engine deliberately distinguishes walk and push actions.  That prevents
the duplicate "push acts as walk" behaviour found in some Gym environments
from leaking into policy labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Final, Iterable, Mapping, TypeAlias

Position: TypeAlias = tuple[int, int]


class Action(str, Enum):
    PUSH_UP = "push_up"
    PUSH_DOWN = "push_down"
    PUSH_LEFT = "push_left"
    PUSH_RIGHT = "push_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"


ACTION_ORDER: Final[tuple[Action, ...]] = tuple(Action)
ACTION_IDS: Final[Mapping[Action, int]] = {
    Action.PUSH_UP: 1,
    Action.PUSH_DOWN: 2,
    Action.PUSH_LEFT: 3,
    Action.PUSH_RIGHT: 4,
    Action.MOVE_UP: 5,
    Action.MOVE_DOWN: 6,
    Action.MOVE_LEFT: 7,
    Action.MOVE_RIGHT: 8,
}
ACTION_DESCRIPTIONS: Final[Mapping[Action, str]] = {
    Action.PUSH_UP: "push the adjacent box one square up",
    Action.PUSH_DOWN: "push the adjacent box one square down",
    Action.PUSH_LEFT: "push the adjacent box one square left",
    Action.PUSH_RIGHT: "push the adjacent box one square right",
    Action.MOVE_UP: "walk one square up without pushing a box",
    Action.MOVE_DOWN: "walk one square down without pushing a box",
    Action.MOVE_LEFT: "walk one square left without pushing a box",
    Action.MOVE_RIGHT: "walk one square right without pushing a box",
}

_DIRECTION: Final[Mapping[Action, Position]] = {
    Action.PUSH_UP: (-1, 0),
    Action.MOVE_UP: (-1, 0),
    Action.PUSH_DOWN: (1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.PUSH_LEFT: (0, -1),
    Action.MOVE_LEFT: (0, -1),
    Action.PUSH_RIGHT: (0, 1),
    Action.MOVE_RIGHT: (0, 1),
}


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


def _is_push(action: Action) -> bool:
    return action.value.startswith("push_")


@dataclass(frozen=True, slots=True)
class Transition:
    """The result of applying one action to a board."""

    board: "Board"
    moved: bool
    pushed: bool


@dataclass(frozen=True, slots=True)
class Board:
    """An immutable Sokoban position.

    Rows and columns are zero-based.  `width` and `height` define the playable
    rectangular canvas; positions outside it are treated as walls.
    """

    width: int
    height: int
    walls: frozenset[Position]
    goals: frozenset[Position]
    boxes: frozenset[Position]
    player: Position

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("board dimensions must be positive")
        all_positions = set(self.walls) | set(self.goals) | set(self.boxes) | {self.player}
        if any(not self.in_bounds(position) for position in all_positions):
            raise ValueError("all board entities must be inside the board")
        if self.player in self.walls or self.player in self.boxes:
            raise ValueError("the player cannot overlap a wall or box")
        if self.walls & self.goals or self.walls & self.boxes:
            raise ValueError("walls cannot overlap goals or boxes")
        if len(self.boxes) != len(self.goals):
            raise ValueError("Sokoban requires the same number of boxes and goals")

    @classmethod
    def from_ascii(cls, text: str | Iterable[str]) -> "Board":
        """Parse ``# . $ @ * +`` Sokoban notation.

        Lines may be ragged.  Missing cells are converted to walls, which
        makes pasted level files safe to parse without allowing escape routes.
        """

        raw_lines = text.splitlines() if isinstance(text, str) else list(text)
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()
        if not raw_lines:
            raise ValueError("level contains no rows")
        width = max(len(line) for line in raw_lines)
        walls: set[Position] = set()
        goals: set[Position] = set()
        boxes: set[Position] = set()
        player: Position | None = None
        allowed = {"#", " ", ".", "$", "@", "*", "+"}
        for row, line in enumerate(raw_lines):
            for col in range(width):
                tile = line[col] if col < len(line) else "#"
                if tile not in allowed:
                    raise ValueError(f"unsupported tile {tile!r} at row {row + 1}, column {col + 1}")
                position = (row, col)
                if tile == "#":
                    walls.add(position)
                elif tile in ".*+":
                    goals.add(position)
                if tile in "$*":
                    boxes.add(position)
                if tile in "@+":
                    if player is not None:
                        raise ValueError("level must contain exactly one player")
                    player = position
        if player is None:
            raise ValueError("level must contain one player (@ or +)")
        return cls(width, len(raw_lines), frozenset(walls), frozenset(goals), frozenset(boxes), player)

    @classmethod
    def from_xsb(cls, text: str | Iterable[str]) -> "Board":
        """Parse an XSB level whose ragged outer whitespace means "void".

        Standard Sokoban level files often indent rows and use spaces outside
        the wall boundary.  A plain ASCII parser cannot distinguish that void
        from walkable floor.  This loader floods boundary whitespace and turns
        only that exterior region into walls before using ``from_ascii``.
        """

        raw_lines = text.splitlines() if isinstance(text, str) else list(text)
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()
        if not raw_lines:
            raise ValueError("level contains no rows")
        width = max(len(line) for line in raw_lines)
        grid = [list(line.ljust(width)) for line in raw_lines]
        height = len(grid)
        exterior: set[Position] = set()
        queue: deque[Position] = deque()
        for row in range(height):
            for col in (0, width - 1):
                if grid[row][col] == " ":
                    queue.append((row, col))
        for col in range(width):
            for row in (0, height - 1):
                if grid[row][col] == " ":
                    queue.append((row, col))
        while queue:
            row, col = queue.popleft()
            position = (row, col)
            if position in exterior or not (0 <= row < height and 0 <= col < width) or grid[row][col] != " ":
                continue
            exterior.add(position)
            queue.extend(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
        for row, col in exterior:
            grid[row][col] = "#"
        return cls.from_ascii("\n".join("".join(row) for row in grid))

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def is_floor(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.walls

    def is_solved(self) -> bool:
        return self.boxes == self.goals

    def render(self) -> str:
        """Render the full, stable-size board in standard ASCII notation."""

        rows: list[str] = []
        for row in range(self.height):
            chars: list[str] = []
            for col in range(self.width):
                position = (row, col)
                if position in self.walls:
                    tile = "#"
                elif position == self.player:
                    tile = "+" if position in self.goals else "@"
                elif position in self.boxes:
                    tile = "*" if position in self.goals else "$"
                elif position in self.goals:
                    tile = "."
                else:
                    tile = " "
                chars.append(tile)
            rows.append("".join(chars))
        return "\n".join(rows)

    def legal_actions(self) -> tuple[Action, ...]:
        """Return non-no-op actions in fixed order.

        A push is legal only when it really pushes a box; this makes every
        action label semantically unique for training.
        """

        return tuple(action for action in ACTION_ORDER if self.apply(action).moved)

    def apply(self, action: Action) -> Transition:
        """Apply an action; invalid actions leave the board unchanged."""

        delta = _DIRECTION[action]
        next_player = _add(self.player, delta)
        if not self.is_floor(next_player):
            return Transition(self, moved=False, pushed=False)

        if next_player not in self.boxes:
            if _is_push(action):
                return Transition(self, moved=False, pushed=False)
            return Transition(
                Board(self.width, self.height, self.walls, self.goals, self.boxes, next_player),
                moved=True,
                pushed=False,
            )

        if not _is_push(action):
            return Transition(self, moved=False, pushed=False)
        next_box = _add(next_player, delta)
        if not self.is_floor(next_box) or next_box in self.boxes:
            return Transition(self, moved=False, pushed=False)
        boxes = set(self.boxes)
        boxes.remove(next_player)
        boxes.add(next_box)
        return Transition(
            Board(self.width, self.height, self.walls, self.goals, frozenset(boxes), next_player),
            moved=True,
            pushed=True,
        )

    def static_dead_squares(self) -> frozenset[Position]:
        """Squares from which a box cannot reach any goal, ignoring boxes.

        This reverse-push analysis is sound: a non-goal box placed on one of
        these tiles is permanently dead.  It is deliberately conservative and
        does not attempt every dynamic deadlock pattern.
        """

        reachable = set(self.goals)
        frontier = list(self.goals)
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        while frontier:
            destination = frontier.pop()
            for delta in directions:
                origin = _add(destination, (-delta[0], -delta[1]))
                player_spot = _add(origin, (-delta[0], -delta[1]))
                if self.is_floor(origin) and self.is_floor(player_spot) and origin not in reachable:
                    reachable.add(origin)
                    frontier.append(origin)
        floor = {
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self.is_floor((row, col))
        }
        return frozenset(floor - reachable)

    def has_static_deadlock(self) -> bool:
        dead = self.static_dead_squares()
        return any(box not in self.goals and box in dead for box in self.boxes)
