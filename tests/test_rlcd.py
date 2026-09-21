from __future__ import annotations

import unittest

from jevgames.contracts import CalibrationDecision, DecisionOption
from jevgames.engine import _reliability_bins
from jevgames.strategies.rlcd import RLCDConfig, group_relative_advantages


class RLCDTests(unittest.TestCase):
    def test_group_relative_advantages_are_normalized_per_decision(self) -> None:
        advantages = group_relative_advantages([
            [1.0, 4.0],
            [2.0, 4.0],
            [3.0, 4.0],
        ])
        self.assertLess(advantages[0][0], 0.0)
        self.assertAlmostEqual(advantages[1][0], 0.0)
        self.assertGreater(advantages[2][0], 0.0)
        self.assertEqual([row[1] for row in advantages], [0.0, 0.0, 0.0])

    def test_calibration_decision_accepts_soft_targets(self) -> None:
        decision = CalibrationDecision(
            serialized_state="state",
            observation={"state": "state"},
            instruction="choose",
            options=(DecisionOption("a", "A"), DecisionOption("b", "B")),
            target_probabilities=(0.75, 0.25),
            evidence_id="sample",
            step=0,
            provenance="environment_outcomes",
        )
        self.assertEqual(decision.target_probabilities, (0.75, 0.25))

    def test_rlcd_rejects_single_sample_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            RLCDConfig(group_size=1)

    def test_calibration_decision_rejects_duplicate_option_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            CalibrationDecision(
                serialized_state="state",
                observation={},
                instruction="choose",
                options=(DecisionOption("a", "A"), DecisionOption("a", "again")),
                target_probabilities=(0.5, 0.5),
                evidence_id="sample",
                step=0,
            )

    def test_reliability_error_tracks_confidence_gap(self) -> None:
        ece, rows = _reliability_bins([0.8, 0.8], [1.0, 0.0], 10)
        self.assertAlmostEqual(ece, 0.3)
        self.assertEqual(rows[0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
