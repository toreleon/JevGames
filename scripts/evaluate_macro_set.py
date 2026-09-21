"""CLI wrapper around the reusable Sokoban benchmark module."""

from __future__ import annotations

import argparse
import json

from sokoban_laya.benchmark import BenchmarkConfig, evaluate_macro_dataset, save_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-levels", type=int)
    parser.add_argument("--max-pushes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = evaluate_macro_dataset(
        args.model,
        args.dataset,
        BenchmarkConfig(args.device, args.max_levels, args.max_pushes, args.batch_size),
        args.manifest,
    )
    print(json.dumps(result, sort_keys=True))
    if args.output:
        save_benchmark(result, args.output)


if __name__ == "__main__":
    main()
