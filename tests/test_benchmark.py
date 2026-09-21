from __future__ import annotations

import unittest

from sokoban_laya.benchmark import _summarize


class BenchmarkTests(unittest.TestCase):
    def test_summary_reports_solve_rate_and_outcomes(self) -> None:
        states = [
            {"reason": "solved", "pushes": 3},
            {"reason": "static_deadlock", "pushes": 2},
        ]
        summary = _summarize(states)
        self.assertEqual(summary["solved"], 1)
        self.assertEqual(summary["solve_rate"], 0.5)
        self.assertEqual(summary["mean_pushes"], 2.5)
        self.assertEqual(summary["outcomes"], {"solved": 1, "static_deadlock": 1})


if __name__ == "__main__":
    unittest.main()
