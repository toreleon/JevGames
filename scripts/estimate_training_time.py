"""Extrapolate M4 training time from a measured macro-GRPO benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def hours(seconds: float) -> float:
    return round(seconds / 3600.0, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_stats")
    parser.add_argument("--benchmark-levels", type=int, required=True)
    parser.add_argument("--pilot-levels", type=int, default=2000)
    parser.add_argument("--target-levels", type=int, default=10000)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--grpo-iterations", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()

    stats = json.loads(Path(args.benchmark_stats).read_text())
    setup = next(row for row in stats if row["phase"] == "setup")
    warmup = next(row for row in stats if row["phase"] in {"macro_warmup", "hf_trainer_macro_warmup"})
    grpo = next(row for row in stats if row["phase"] == "environment_grpo")
    fixed_setup = setup["seconds"]
    macro_per_level = setup["trainable_macro_steps"] / args.benchmark_levels
    warmup_examples_per_second = setup["trainable_macro_steps"] / warmup["seconds"]
    grpo_episodes_per_second = grpo["episodes"] / grpo["seconds"]

    estimates = []
    for name, levels in (("pilot", args.pilot_levels), ("target", args.target_levels)):
        macro_examples = levels * macro_per_level
        warmup_seconds = macro_examples * args.warmup_epochs / warmup_examples_per_second
        episodes = levels * args.group_size * args.grpo_iterations
        grpo_seconds = episodes / grpo_episodes_per_second
        total_seconds = fixed_setup + warmup_seconds + grpo_seconds
        estimates.append({
            "name": name,
            "levels": levels,
            "estimated_macro_examples": round(macro_examples),
            "warmup_epochs": args.warmup_epochs,
            "grpo_episodes": episodes,
            "warmup_hours": hours(warmup_seconds),
            "grpo_hours": hours(grpo_seconds),
            "total_hours": hours(total_seconds),
        })
    report = {
        "measured": {
            "hardware": "Apple M4 Pro 48 GB",
            "benchmark_levels": args.benchmark_levels,
            "warmup_examples_per_second": round(warmup_examples_per_second, 3),
            "grpo_episodes_per_second": round(grpo_episodes_per_second, 3),
            "setup_seconds": fixed_setup,
        },
        "assumptions": {
            "linear_scaling": True,
            "batch_size": 16,
            "rollout_inference_batch_size": 32,
            "encoder_frozen": True,
            "uncertainty": "±25%; longer levels and thermal throttling can increase runtime",
        },
        "estimates": estimates,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
