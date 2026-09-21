import copy
import unittest

from jevgames.research.choice_inference import prepare_request, trial, validate_answer
from sokoban_laya.core import Board
from sokoban_laya.macros import legal_push_macros, apply_push_macro
from jevgames.research.bounded_sokoban import successor
from scripts.summarize_choice_inference import audit_trial


class ChoiceInferenceTests(unittest.TestCase):
    def setUp(self):
        self.board = Board.from_ascii("#####\n# . #\n# $ #\n# @ #\n#####")

    def test_presentations_have_identical_executable_options(self):
        state_a, question_a, actions_a = prepare_request(self.board, [], 8, "ascii", 17, "board", 0)
        state_b, question_b, actions_b = prepare_request(self.board, [], 8, "structured", 17, "board", 0)
        self.assertEqual(question_a, question_b)
        self.assertEqual(actions_a, actions_b)
        self.assertEqual({a.label for a in actions_a.values()}, {a.label for a in legal_push_macros(self.board)})
        self.assertEqual(state_b["player_rc"], [4, 3])
        self.assertNotIn("oracle", str(state_b))
        self.assertEqual(state_a["board"], state_b["board"])
        for action in actions_a.values():
            child, _ = apply_push_macro(self.board, action)
            self.assertEqual(child, successor(self.board, action))

    def test_option_shuffle_never_changes_action_semantics(self):
        orders = []
        for seed in (17, 29, 43):
            _, q, actions = prepare_request(self.board, [], 8, "structured", seed, "board", 0)
            orders.append(tuple(a.label for a in actions.values()))
            for key, macro in actions.items():
                self.assertEqual(q["criteria"][key], f"r{macro.box[0]+1}c{macro.box[1]+1} {macro.direction}")
        self.assertGreater(len(set(orders)), 1)

    def test_invalid_or_mismatched_native_output_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_answer({"choice": "Z", "probabilities": {"A": 1}}, ("A", "B"))
        with self.assertRaises(ValueError):
            validate_answer({"choice": "A", "probabilities": {"A": .1, "B": .9}}, ("A", "B"))
        validate_answer({"choice": "B", "probabilities": {"A": .1, "B": .9}}, ("A", "B"))

    def test_trial_audit_replays_primitive_actions_and_rejects_changed_trace(self):
        level = {"layout_id": "example", "board": self.board.render(),
                 "difficulty": "test", "initial_action_costs": None, "optimal_pushes": None}
        report = trial([level], None, "random", "ascii", 17, 8)
        audit_trial(report, {"example": level}, 8)
        damaged = copy.deepcopy(report)
        damaged["results"][0]["trace"][0]["selected_macro"] = "not_the_selected_action"
        with self.assertRaises(ValueError):
            audit_trial(damaged, {"example": level}, 8)

    def test_trial_audit_rejects_wrong_outcome(self):
        level = {"layout_id": "example", "board": self.board.render(),
                 "difficulty": "test", "initial_action_costs": None, "optimal_pushes": None}
        report = trial([level], None, "random", "structured", 29, 8)
        report["results"][0]["outcome"] = "made_up_outcome"
        with self.assertRaises(ValueError):
            audit_trial(report, {"example": level}, 8)


if __name__ == "__main__":
    unittest.main()
