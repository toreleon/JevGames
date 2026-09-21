"""Create a tiny, deterministic solver-labelled set for an end-to-end smoke run.

This is intentionally not a benchmark or a useful production training corpus.
It merely proves that the level parser, solver, JSONL exporter, Laya trainer,
and MPS inference controller agree on their action semantics.
"""

from sokoban_laya.core import Board
from sokoban_laya.solver import solve
from sokoban_laya.trajectories import solution_trajectory, write_jsonl


LEVELS = {
    "right_one": """\
#####
#@$.#
#####
""",
    "right_two": """\
########
#@ $ . #
########
""",
    "left_two": """\
########
# . $ @#
########
""",
    "up": """\
#######
#  .  #
#  $  #
#  @  #
#######
""",
    "down": """\
#######
#  @  #
#  $  #
#  .  #
#######
""",
    "walk_right_then_push": """\
#######
#  .  #
#  $  #
# @   #
#######
""",
    "walk_left_then_push": """\
#######
#  .  #
#  $  #
#   @ #
#######
""",
    "two_boxes": """\
#########
# .   . #
# $   $ #
# @     #
#########
""",
}


def main() -> None:
    examples = []
    for name, level in LEVELS.items():
        board = Board.from_ascii(level)
        result = solve(board)
        if result is None:
            raise RuntimeError(f"smoke level {name!r} unexpectedly has no solution")
        examples.extend(solution_trajectory(board, result, name))
        print(f"{name:24} {len(result.actions):2} actions, {result.pushes:2} pushes")
    write_jsonl(examples, "data/smoke_train.jsonl")
    print(f"Wrote {len(examples)} training decisions to data/smoke_train.jsonl")


if __name__ == "__main__":
    main()
