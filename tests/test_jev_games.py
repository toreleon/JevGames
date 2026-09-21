from __future__ import annotations

import unittest

from jevgames.config import load_experiment_config
from jevgames.registry import available_plugins, create_task
from sokoban_laya.core import Board


class JevGamesFrameworkTests(unittest.TestCase):
    def test_builtin_plugins_are_discoverable(self) -> None:
        self.assertEqual(available_plugins(), {"models": ["laya"], "tasks": ["sokoban_push"]})

    def test_config_selects_model_and_task_independently(self) -> None:
        config = load_experiment_config("configs/sokoban_smoke.toml")
        self.assertEqual(config.model_type, "laya")
        self.assertEqual(config.task_type, "sokoban_push")
        self.assertEqual(config.train_dataset, "data/smoke_train.jsonl")
        self.assertEqual(config.online.learning_rate, 3e-4)
        self.assertEqual(config.online.gradient_accumulation, 2)
        self.assertEqual(config.warmup.max_grad_norm, 1.0)

    def test_sokoban_task_is_accessed_only_through_contract(self) -> None:
        task = create_task("sokoban_push", {})
        board = Board.from_ascii("#######\n#  .  #\n#  $  #\n# @   #\n#######")
        options = task.options(board)
        self.assertGreaterEqual(len(options), 1)
        outcome = task.transition(board, options[0].key)
        self.assertGreaterEqual(outcome.primitive_steps, 1)


if __name__ == "__main__":
    unittest.main()
