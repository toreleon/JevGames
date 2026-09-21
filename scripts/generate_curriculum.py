"""Generate leak-free train/validation/test Sokoban curricula."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from sokoban_laya.generation import difficulty_schedule, generate_level
from sokoban_laya.solver import SolveResult
from sokoban_laya.trajectories import solution_trajectory, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pilot")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    used_layouts: set[str] = set()
    used_levels: set[str] = set()
    manifest = {"seed": args.seed, "splits": {}}
    started = time.perf_counter()
    seed_cursor = args.seed

    for split, count in (("train", args.train), ("validation", args.validation), ("test", args.test)):
        examples = []
        records = []
        for difficulty in difficulty_schedule(count):
            while True:
                level = generate_level(difficulty, seed_cursor)
                seed_cursor += 1
                if level.layout_id not in used_layouts and level.level_id not in used_levels:
                    break
            used_layouts.add(level.layout_id)
            used_levels.add(level.level_id)
            episode = f"{split}_{len(records):06d}_{level.level_id}"
            trajectory = solution_trajectory(level.board, SolveResult(level.solution, 0), episode)
            examples.extend(trajectory)
            records.append({
                "episode": episode,
                "layout_id": level.layout_id,
                "level_id": level.level_id,
                "difficulty": level.difficulty,
                "seed": level.seed,
                "width": level.board.width,
                "height": level.board.height,
                "boxes": len(level.board.boxes),
                "reverse_pushes": level.reverse_pushes,
                "expert_actions": len(level.solution),
                "expert_pushes": sum(action.value.startswith("push_") for action in level.solution),
            })
        write_jsonl(examples, destination / f"{split}.jsonl")
        manifest["splits"][split] = {
            "levels": len(records),
            "decisions": len(examples),
            "records": records,
        }
        print(json.dumps({"split": split, "levels": len(records), "decisions": len(examples)}), flush=True)

    manifest["unique_layouts"] = len(used_layouts)
    manifest["unique_levels"] = len(used_levels)
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({
        "output": str(destination),
        "unique_layouts": len(used_layouts),
        "unique_levels": len(used_levels),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
