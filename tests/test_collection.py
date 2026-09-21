from __future__ import annotations

import random
import unittest

from jevgames.collectors.episodic import EpisodicCollectorConfig, _sample_query
from jevgames.contracts import ActionOutcomeQuery, DecisionKind, DecisionOption, DecisionQuestion


def _query(key: str) -> ActionOutcomeQuery:
    return ActionOutcomeQuery(
        key,
        DecisionQuestion(
            key,
            DecisionKind.NOUL,
            key,
            (DecisionOption("false", "failure"), DecisionOption("true", "success")),
        ),
    )


class CollectionTests(unittest.TestCase):
    def test_full_epsilon_explores_low_scored_actions(self) -> None:
        rng = random.Random(7)
        candidates = ((_query("high"), 0.99), (_query("low"), 0.01))
        selected = {
            _sample_query(candidates, temperature=1.0, epsilon=1.0, rng=rng).action_key
            for _ in range(20)
        }
        self.assertEqual(selected, {"high", "low"})

    def test_invalid_exploration_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "epsilon"):
            EpisodicCollectorConfig(epsilon=1.1)


if __name__ == "__main__":
    unittest.main()
