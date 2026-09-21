"""Public Jev Games command-line interface."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import load_experiment_config
from .experiment import run_experiment
from .registry import available_plugins


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-games", description="Pluggable decision-policy training and benchmark framework")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run a TOML experiment")
    run_parser.add_argument("config")
    commands.add_parser("plugins", help="list registered model and task plugins")
    args = parser.parse_args(argv)
    if args.command == "plugins":
        print(json.dumps(available_plugins(), indent=2))
        return 0
    report = run_experiment(load_experiment_config(args.config))
    print(json.dumps({
        "name": report["name"],
        "checkpoint": report["checkpoint"],
        "benchmarks": report["benchmarks"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
