from __future__ import annotations

import unittest

from sokoban_laya.generation import Difficulty, generate_level
from sokoban_laya.solver import replay


class GenerationTests(unittest.TestCase):
    def test_reverse_generated_solution_replays(self) -> None:
        difficulty = Difficulty("test", 7, 7, 2, 4, 6, 0.05)
        level = generate_level(difficulty, seed=1234)
        self.assertFalse(level.board.is_solved())
        self.assertTrue(replay(level.board, level.solution).is_solved())
        self.assertGreaterEqual(level.reverse_pushes, 4)

    def test_generation_is_deterministic(self) -> None:
        difficulty = Difficulty("test", 7, 7, 2, 4, 6, 0.05)
        first = generate_level(difficulty, seed=99)
        second = generate_level(difficulty, seed=99)
        self.assertEqual(first.board, second.board)
        self.assertEqual(first.solution, second.solution)


if __name__ == "__main__":
    unittest.main()
