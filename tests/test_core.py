from __future__ import annotations

import unittest

from sokoban_laya.core import Action, Board


class BoardTests(unittest.TestCase):
    def test_parse_render_and_solve_one_push_level(self) -> None:
        board = Board.from_ascii("#####\n#@$.#\n#####")
        self.assertEqual(board.render(), "#####\n#@$.#\n#####")
        self.assertEqual(board.legal_actions(), (Action.PUSH_RIGHT,))

        transition = board.apply(Action.PUSH_RIGHT)
        self.assertTrue(transition.moved)
        self.assertTrue(transition.pushed)
        self.assertTrue(transition.board.is_solved())
        self.assertEqual(transition.board.render(), "#####\n# @*#\n#####")

    def test_walk_cannot_push_and_invalid_actions_are_noops(self) -> None:
        board = Board.from_ascii("#####\n#@$.#\n#####")
        blocked = board.apply(Action.MOVE_RIGHT)
        self.assertFalse(blocked.moved)
        self.assertIs(blocked.board, board)

    def test_static_dead_square_detects_non_goal_corner(self) -> None:
        board = Board.from_ascii("#######\n#@    #\n#$  . #\n#######")
        self.assertTrue(board.has_static_deadlock())

    def test_xsb_parser_turns_outer_whitespace_into_walls(self) -> None:
        level = "    ####\n###  ####\n#     $ #\n# #  #$ #\n# . .#@ #\n#########"
        board = Board.from_xsb(level)
        self.assertEqual(len(board.boxes), 2)
        self.assertEqual(len(board.goals), 2)
        self.assertIn((0, 0), board.walls)
        self.assertIn((0, board.width - 1), board.walls)


if __name__ == "__main__":
    unittest.main()
