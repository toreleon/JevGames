from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokoban_laya.core import Action, Board
from sokoban_laya.laya_policy import LayaPolicy
from sokoban_laya.solver import replay, solve
from sokoban_laya.trajectories import read_jsonl, solution_trajectory, write_jsonl


class FakeLaya:
    def predict(self, state, questions):
        self.state = state
        self.questions = questions
        self.called = True
        choice = next(iter(questions["next_action"]["criteria"]))
        return {
            "answers": {
                "next_action": {
                    "choice": choice,
                    "probabilities": {choice: 1.0},
                    "confidence": 0.99,
                }
            }
        }


class SolverAndPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board.from_ascii("#####\n#@$.#\n#####")

    def test_solver_solution_replays_to_a_win(self) -> None:
        result = solve(self.board)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.actions, (Action.PUSH_RIGHT,))
        self.assertTrue(replay(self.board, result.actions).is_solved())

    def test_solver_walks_before_pushing(self) -> None:
        board = Board.from_ascii("#######\n#  .  #\n#  $  #\n# @   #\n#######")
        result = solve(board)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.actions, (Action.MOVE_RIGHT, Action.PUSH_UP))
        self.assertTrue(replay(board, result.actions).is_solved())

    def test_trajectory_round_trip(self) -> None:
        result = solve(self.board)
        assert result is not None
        examples = solution_trajectory(self.board, result, "tiny")
        self.assertEqual(examples[0].expert_action, "push_right")
        self.assertEqual(examples[0].legal_actions, ("push_right",))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "data.jsonl"
            write_jsonl(examples, output)
            self.assertEqual(read_jsonl(output), examples)
            self.assertEqual(json.loads(output.read_text())["board"], self.board.render())

    def test_policy_uses_only_legal_choices(self) -> None:
        fake = FakeLaya()
        board = Board.from_ascii("#######\n#  .  #\n#  $  #\n# @   #\n#######")
        decision = LayaPolicy(fake).choose(board)
        self.assertEqual(decision.action, Action.MOVE_UP)
        self.assertEqual(fake.state["board"], board.render())
        self.assertEqual(tuple(fake.questions["next_action"]["criteria"]), ("move_up", "move_left", "move_right"))

    def test_policy_executes_forced_move_without_model(self) -> None:
        fake = FakeLaya()
        fake.called = False
        decision = LayaPolicy(fake).choose(self.board)
        self.assertEqual(decision.action, Action.PUSH_RIGHT)
        self.assertFalse(fake.called)


if __name__ == "__main__":
    unittest.main()
