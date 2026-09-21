"""Run a model/task experiment selected entirely by TOML configuration."""

from __future__ import annotations

import argparse
import json

from jevgames.config import load_experiment_config
from jevgames.experiment import run_experiment
from jevgames.registry import available_plugins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?")
    parser.add_argument("--list-plugins", action="store_true")
    args = parser.parse_args()
    if args.list_plugins:
        print(json.dumps(available_plugins(), indent=2))
        return
    if not args.config:
        parser.error("config is required unless --list-plugins is used")
    report = run_experiment(load_experiment_config(args.config))
    print(json.dumps({
        "name": report["name"],
        "checkpoint": report["checkpoint"],
        "benchmarks": report["benchmarks"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
