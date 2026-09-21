from __future__ import annotations

from collections import deque
import importlib.util
import unittest

from jevgames.research.bounded_sokoban import exact_graph, state_key, successor, question
from sokoban_laya.core import Board
from sokoban_laya.generation import Difficulty, generate_level
from sokoban_laya.macros import apply_push_macro, legal_push_macros


class ExactPushOracleTests(unittest.TestCase):
    def test_macro_successor_and_push_distances_match_primitive_engine(self):
        level = generate_level(Difficulty("oracle", 6, 6, 2, 2, 4, 0.03), 918)
        graph = exact_graph(level.board)
        # Independent shortest-push search retains exact player position and
        # executes every walking+push primitive, rather than region merging.
        frontier = deque([(level.board, 0)])
        seen = {level.board}
        shortest = None
        while frontier:
            board, distance = frontier.popleft()
            if board.is_solved():
                shortest = distance
                break
            for macro in legal_push_macros(board):
                child, _ = apply_push_macro(board, macro)
                self.assertEqual(child, successor(board, macro))
                if child not in seen:
                    seen.add(child)
                    frontier.append((child, distance + 1))
        self.assertEqual(graph.distances[state_key(level.board)], shortest)

    def test_horizon_counts_first_push_and_distinguishes_deadlock(self):
        board = Board.from_ascii("#####\n# . #\n# $ #\n# @ #\n#####")
        graph = exact_graph(board)
        self.assertEqual(graph.distances[state_key(board)], 1)
        costs = {action: graph.distances.get(child) for action, child in graph.edges[state_key(board)]}
        self.assertEqual(costs["box_r2_c2_push_up"], 0)
        self.assertIsNone(costs["box_r2_c2_push_left"])
        self.assertIn("Including this first push", question("push up", 1).instruction)
        with self.assertRaises(ValueError):
            question("push up", 0)

    def test_budget_exhaustion_is_not_a_negative_label(self):
        board = Board.from_ascii("#####\n# . #\n# $ #\n# @ #\n#####")
        with self.assertRaises(OverflowError):
            exact_graph(board, max_states=1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class ObjectiveControlTests(unittest.TestCase):
    def test_both_objectives_move_probability_towards_target(self):
        import torch
        from jevgames.strategies.rlcd import DirectProperScoreStrategy, RLCDStrategy

        for cls in (DirectProperScoreStrategy, RLCDStrategy):
            logits = torch.zeros(2, 2, requires_grad=True)
            targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            strategy = cls({"group_size": 256})
            generator = torch.Generator().manual_seed(91)
            loss, _, _ = strategy.objective(
                logits, targets, torch.ones_like(targets, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool), 0.3, generator,
            )
            loss.backward()
            updated = logits.detach() - 0.1 * logits.grad
            probabilities = torch.softmax(updated, -1)
            self.assertGreater(float(probabilities[0, 0]), 0.5)
            self.assertGreater(float(probabilities[1, 1]), 0.5)

    def test_rlcd_sampling_does_not_change_dropout_rng(self):
        import torch
        from jevgames.strategies.rlcd import RLCDStrategy

        state = torch.random.get_rng_state().clone()
        logits = torch.zeros(2, 2, requires_grad=True)
        RLCDStrategy({}).objective(
            logits, torch.tensor([[1., 0.], [0., 1.]]),
            torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            0.3, torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
