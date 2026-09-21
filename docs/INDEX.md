# Jev Games documentation

This documentation describes Jev Games as a general decision-model training
framework. Laya and Sokoban are used as concrete examples, not assumed by the
core engine.

## Start here

1. [Getting started](GETTING_STARTED.md) — install, verify, and run the smoke
   experiment.
2. [Architecture](ARCHITECTURE.md) — understand model/task separation and the
   experiment lifecycle.
3. [Configuration reference](CONFIGURATION.md) — every supported TOML field.
4. [Training semantics](TRAINING.md) — expert warm-up, online GRPO, rewards,
   masking, and reproducibility.
5. [Benchmarking protocol](BENCHMARKING.md) — held-out evaluation and report
   interpretation.

## Extend and operate

- [Plugin API](PLUGINS.md) — implement another model or task.
- [Artifacts and schemas](ARTIFACTS.md) — checkpoints, reports, datasets, and
  manifests.
- [Operations and scaling](OPERATIONS.md) — MPS, CUDA, resource planning,
  performance tuning, and troubleshooting.
- [Command-line interface](CLI.md) — public commands and compatibility tools.
- [Development guide](DEVELOPMENT.md) — repository workflow and verification.
- [Framework selection](FRAMEWORKS.md) — why Jev Games uses Trainer,
  Accelerate, and TorchRL rather than forcing Laya through an LLM trainer.

## Case studies

- [Sokoban](SOKOBAN.md) — reverse-play generation, push macros, reward shaping,
  measured results, and known limits.

## API source of truth

The authoritative interfaces live in:

- [`jevgames/contracts.py`](../jevgames/contracts.py)
- [`jevgames/config.py`](../jevgames/config.py)
- [`jevgames/engine.py`](../jevgames/engine.py)
- [`jevgames/registry.py`](../jevgames/registry.py)

Documentation should be updated whenever those public contracts change.
