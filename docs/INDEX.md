# Jev Games documentation

Jev Games is a pluggable calibrated-decision training framework. Laya and
Sokoban are built-in integrations rather than core assumptions.

## Start here

1. [Getting started](GETTING_STARTED.md) — install, generate data, and run RLCD.
2. [Architecture](ARCHITECTURE.md) — model, task, evidence, collector, and strategy boundaries.
3. [RLCD training](TRAINING.md) — evidence, exploration, scoring, and updates.
4. [Configuration](CONFIGURATION.md) — supported TOML fields.
5. [Benchmarking](BENCHMARKING.md) — calibration and environment protocols.

## Extend and operate

- [Plugin API](PLUGINS.md)
- [Artifacts and schemas](ARTIFACTS.md)
- [Operations and scaling](OPERATIONS.md)
- [CLI](CLI.md)
- [Framework selection](FRAMEWORKS.md)
- [Design sources](DESIGN_SOURCES.md)
- [Development](DEVELOPMENT.md)

## Case study

- [Sokoban](SOKOBAN.md)
- [RLCD objective comparison](OBJECTIVE_ABLATION.md) — registered paired study.

The public contracts in `jevgames/contracts.py`, `config.py`, `registry.py`,
and `engine.py` are the source of truth.
