from __future__ import annotations

import unittest

from sokoban_laya.grpo_training import EpisodeResult, RolloutStep, _group_training_rows, initial_boards
from sokoban_laya.trajectories import TrainingExample


class GRPOHelpersTests(unittest.TestCase):
    def test_group_advantages_prefer_high_return_episode(self) -> None:
        decision = RolloutStep("#####\n#@$.#\n#####", ("move_left", "move_right"), 1, -0.69)
        episodes = [
            EpisodeResult(10.0, True, 1, 2, "solved", (decision,)),
            EpisodeResult(-2.0, False, 0, 2, "repeat", (decision,)),
        ]
        rows = _group_training_rows(episodes)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0][1], 0)
        self.assertLess(rows[1][1], 0)

    def test_equal_returns_produce_no_grpo_update(self) -> None:
        episodes = [EpisodeResult(0.0, False, 0, 1, "limit", ()) for _ in range(4)]
        self.assertEqual(_group_training_rows(episodes), [])

    def test_initial_boards_selects_episode_step_zero(self) -> None:
        examples = [
            TrainingExample("#####\n#@$.#\n#####", ("push_right",), "push_right", "one", 0),
            TrainingExample("#####\n# @*#\n#####", (), "push_right", "one", 1),
        ]
        boards = initial_boards(examples)
        self.assertEqual(len(boards), 1)
        self.assertEqual(boards[0].render(), examples[0].board)


if __name__ == "__main__":
    unittest.main()
