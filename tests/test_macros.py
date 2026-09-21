from __future__ import annotations

import unittest

from sokoban_laya.core import Action, Board
from sokoban_laya.macro_grpo_training import compress_expert_trajectories
from sokoban_laya.macros import apply_push_macro, legal_push_macros
from sokoban_laya.solver import solve
from sokoban_laya.trajectories import solution_trajectory


class PushMacroTests(unittest.TestCase):
    def test_macro_walks_then_pushes(self) -> None:
        board = Board.from_ascii("#######\n#  .  #\n#  $  #\n# @   #\n#######")
        macros = legal_push_macros(board)
        push_up = next(macro for macro in macros if macro.direction == "up")
        result, primitives = apply_push_macro(board, push_up)
        self.assertEqual(primitives, (Action.MOVE_RIGHT, Action.PUSH_UP))
        self.assertTrue(result.is_solved())

    def test_solver_trajectory_compresses_to_push_count(self) -> None:
        board = Board.from_ascii("#########\n# .   . #\n# $   $ #\n# @     #\n#########")
        solution = solve(board)
        self.assertIsNotNone(solution)
        assert solution is not None
        primitive = solution_trajectory(board, solution, "two")
        macro = compress_expert_trajectories(primitive)
        self.assertEqual(len(macro), solution.pushes)
        self.assertEqual(len(macro), 2)


if __name__ == "__main__":
    unittest.main()
