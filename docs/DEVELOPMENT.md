# Development guide

## Environment

```bash
uv sync --extra laya --extra train --python 3.12
```

Do not add undeclared packages directly to `.venv`. Update `pyproject.toml` and
refresh `uv.lock` through `uv`.

## Verification

Run before handing off changes:

```bash
uv run --extra laya --extra train python -m unittest discover -s tests -v
uv run --extra laya --extra train python -m py_compile \
  jevgames/*.py jevgames/models/*.py jevgames/tasks/*.py \
  sokoban_laya/*.py scripts/*.py
uv lock --check
```

For training-path changes, also run:

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

Compilation and unit tests do not exercise MPS model forward/backward behavior.

## Change boundaries

- generic orchestration belongs in `jevgames`;
- model-specific code belongs in `jevgames/models` or an external adapter;
- task/dataset semantics belong in `jevgames/tasks` or an external adapter;
- Sokoban engine and generation code belong in `sokoban_laya`;
- reproducible experiment parameters belong in `configs`;
- one-off analysis should become a script only when it is reproducible and
  documented.

Avoid adding model or task conditionals to `jevgames.engine`. A conditional on
`model_type` or `task_type` is normally evidence that the adapter contract is
missing a capability.

## Adding configuration fields

Framework fields require:

1. a typed dataclass field in `jevgames.config`;
2. an entry in `docs/CONFIGURATION.md`;
3. a test covering parsing/default behavior;
4. engine use that preserves old defaults.

Plugin-specific fields belong in the plugin table and should be validated by
the plugin factory.

## Adding report fields

Additive fields may remain under schema version 1. Breaking changes require:

- incrementing `schema_version`;
- documenting old and new meanings;
- updating benchmark/report consumers;
- adding a migration note.

## Test strategy

Use layers:

- pure unit tests for environment rules, rewards, registries, and config;
- adapter tests for encoding, masks, loss, save/reload;
- framework tests with lightweight fake model/task plugins;
- smoke integration with the real built-in model and task;
- held-out benchmark for behavior.

Do not substitute training-set success for integration or benchmark coverage.

## Backwards compatibility

`sokoban_laya` is a compatibility implementation used by the built-in plugins
and earlier experiments. Avoid changing its checkpoint or dataset behavior
without verifying both the legacy CLI and Jev Games adapter.

## Generated files

Do not commit or review generated model weights, caches, virtual environments,
or `__pycache__` as source. Reports and small manifests may be retained when
they document a reproducible benchmark.
