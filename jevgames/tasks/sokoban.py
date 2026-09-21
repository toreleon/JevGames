"""Sokoban push-macro task adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Mapping

from jevgames.contracts import DecisionOption, DecisionTask, ExpertDecision, TaskTransition
from jevgames.registry import register_task
from sokoban_laya.core import Board
from sokoban_laya.grpo_training import initial_boards
from sokoban_laya.laya_policy import BOARD_LEGEND
from sokoban_laya.macro_grpo_training import compress_expert_trajectories
from sokoban_laya.macros import apply_push_macro, legal_push_macros, reachable_walk_paths
from sokoban_laya.trajectories import read_jsonl


class SokobanPushTask(DecisionTask):
    name = "sokoban_push"
    instruction = (
        "Choose the reachable box push most likely to solve the Sokoban level. "
        "Avoid irreversible deadlocks and preserve access to every remaining box."
    )

    def __init__(self, config: dict[str, Any]) -> None:
        self.solve_reward = float(config.get("solve_reward", 50.0))
        self.goal_reward = float(config.get("goal_reward", 4.0))
        self.off_goal_penalty = float(config.get("off_goal_penalty", 4.0))
        self.push_penalty = float(config.get("push_penalty", 0.02))
        self.walk_penalty = float(config.get("walk_penalty", 0.002))
        self.deadlock_penalty = float(config.get("deadlock_penalty", 15.0))
        self._repeat_penalty = float(config.get("repeat_penalty", 4.0))
        self._limit_penalty = float(config.get("limit_penalty", 1.0))

    def initial_states(self, dataset_path: str | Path, limit: int | None = None) -> list[Board]:
        return initial_boards(read_jsonl(dataset_path), limit)

    def expert_decisions(self, dataset_path: str | Path) -> list[ExpertDecision]:
        primitive = read_jsonl(dataset_path)
        macro_examples = compress_expert_trajectories(primitive)
        decisions = []
        for example in macro_examples:
            board = Board.from_ascii(example.board)
            options = self.options(board)
            decisions.append(
                ExpertDecision(
                    serialized_state=example.board,
                    observation=self.observation(board),
                    instruction=self.instruction,
                    options=options,
                    target_key=example.expert_label,
                    episode_id=example.episode,
                    step=example.push_step,
                )
            )
        return decisions

    def observation(self, state: Board) -> Mapping[str, Any]:
        return {"board": state.render(), "legend": BOARD_LEGEND}

    def options(self, state: Board) -> tuple[DecisionOption, ...]:
        return tuple(DecisionOption(macro.label, macro.description) for macro in legal_push_macros(state))

    def transition(self, state: Board, option_key: str) -> TaskTransition:
        macros = {macro.label: macro for macro in legal_push_macros(state)}
        if option_key not in macros:
            raise ValueError(f"unknown or illegal action {option_key!r}")
        before_goals = len(state.boxes & state.goals)
        board, primitives = apply_push_macro(state, macros[option_key])
        after_goals = len(board.boxes & board.goals)
        delta = after_goals - before_goals
        reward = -self.push_penalty - self.walk_penalty * max(0, len(primitives) - 1)
        reward += max(0, delta) * self.goal_reward
        reward -= max(0, -delta) * self.off_goal_penalty
        if board.is_solved():
            reward += self.solve_reward
            return TaskTransition(board, reward, True, True, "solved", len(primitives))
        if board.has_static_deadlock():
            reward -= self.deadlock_penalty
            return TaskTransition(board, reward, True, False, "static_deadlock", len(primitives))
        return TaskTransition(board, reward, False, False, None, len(primitives))

    def serialize_state(self, state: Board) -> str:
        return state.render()

    def deserialize_state(self, value: str) -> Board:
        return Board.from_ascii(value)

    def state_key(self, state: Board) -> Hashable:
        return state.boxes, frozenset(reachable_walk_paths(state))

    def is_solved(self, state: Board) -> bool:
        return state.is_solved()

    def repeated_state_reward(self) -> float:
        return -self._repeat_penalty

    def no_action_reward(self) -> float:
        return -self.deadlock_penalty

    def limit_reward(self) -> float:
        return -self._limit_penalty


@register_task("sokoban_push")
def create_sokoban_push_task(config: dict[str, Any]) -> SokobanPushTask:
    return SokobanPushTask(config)
