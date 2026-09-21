from __future__ import annotations

import importlib.util
import unittest

from jevgames.contracts import (
    CalibrationDecision,
    DecisionKind,
    DecisionOption,
    DecisionQuestion,
)
from jevgames.engine import _reliability_bins, _selective_accuracy
from jevgames.scoring import composite_proper_score
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
            question=DecisionQuestion(
                "route",
                DecisionKind.CHOICE,
                "choose",
                (DecisionOption("a", "A"), DecisionOption("b", "B")),
            ),
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
            DecisionQuestion(
                "route",
                DecisionKind.CHOICE,
                "choose",
                (DecisionOption("a", "A"), DecisionOption("a", "again")),
            )

    def test_noul_has_fixed_false_true_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "false, true"):
            DecisionQuestion(
                "safe",
                DecisionKind.NOUL,
                "Is it safe?",
                (DecisionOption("true", "yes"), DecisionOption("false", "no")),
            )

    def test_reliability_error_tracks_confidence_gap(self) -> None:
        ece, rows = _reliability_bins([0.8, 0.8], [1.0, 0.0], 10)
        self.assertAlmostEqual(ece, 0.3)
        self.assertEqual(rows[0]["count"], 2)

    def test_selective_accuracy_ranks_entropy_confidence(self) -> None:
        rows = _selective_accuracy([0.1, 0.9, 0.8, 0.2], [0.0, 1.0, 1.0, 0.0])
        self.assertEqual(rows[-1]["coverage"], 0.5)
        self.assertEqual(rows[-1]["accuracy"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional test dependency")
    def test_ranked_probability_score_respects_ordinal_distance(self) -> None:
        import torch

        probabilities = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        targets = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        scores = composite_proper_score(probabilities, targets, mask, torch.tensor([True, True]))
        self.assertGreater(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()
