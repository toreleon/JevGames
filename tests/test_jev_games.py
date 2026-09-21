from __future__ import annotations

import unittest

from jevgames.config import load_experiment_config
from jevgames.registry import available_plugins, create_task
from sokoban_laya.core import Board


class JevGamesFrameworkTests(unittest.TestCase):
    def test_builtin_plugins_are_discoverable(self) -> None:
        self.assertEqual(available_plugins(), {
            "collectors": ["episodic_outcomes"],
            "evidence": ["sokoban_solver"],
            "models": ["laya", "qwen_decision"],
            "strategies": ["rlcd"],
            "tasks": ["sokoban_push"],
        })

    def test_config_selects_model_and_task_independently(self) -> None:
        config = load_experiment_config("configs/sokoban_smoke.toml")
        self.assertEqual(config.model_type, "laya")
        self.assertEqual(config.task_type, "sokoban_push")
        self.assertEqual(config.training_type, "rlcd")
        self.assertEqual(config.evidence_type, "sokoban_solver")
        self.assertEqual(config.train_dataset, "data/smoke_train.jsonl")
        self.assertEqual(config.calibration_dataset, "data/pilot100/calibration.jsonl")
        self.assertEqual(config.collector_type, "episodic_outcomes")
        self.assertEqual(config.training["learning_rate"], 3e-4)
        self.assertEqual(config.training["gradient_accumulation"], 2)
        self.assertEqual(config.training["group_size"], 4)

    def test_sokoban_task_is_accessed_only_through_contract(self) -> None:
        task = create_task("sokoban_push", {})
        board = Board.from_ascii("#######\n#  .  #\n#  $  #\n# @   #\n#######")
        options = task.question(board).options
        self.assertGreaterEqual(len(options), 1)
        outcome_queries = task.outcome_queries(board)
        self.assertEqual({query.action_key for query in outcome_queries}, {option.key for option in options})
        self.assertTrue(all(query.question.kind.value == "noul" for query in outcome_queries))
        outcome = task.transition(board, options[0].key)
        self.assertGreaterEqual(outcome.primitive_steps, 1)


if __name__ == "__main__":
    unittest.main()
