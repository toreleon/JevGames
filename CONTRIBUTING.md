# Contributing to Jev Games

## Before opening a change

- Discuss large contract, schema, or training-algorithm changes in an issue.
- Keep the generic engine independent of individual models and tasks.
- Put model-specific behavior in a model adapter.
- Put environment and transition behavior in a task adapter.
- Put dataset parsing and target provenance in an evidence provider.
- Put optimization algorithms in a training strategy.
- Add optional dependencies to an appropriate project extra.

## Development setup

```bash
uv sync --extra laya --extra train --python 3.12
```

## Required verification

```bash
uv run --extra laya --extra train python -m unittest discover -s tests -v
uv run --extra laya --extra train python -m py_compile \
  jevgames/*.py jevgames/evidence/*.py jevgames/models/*.py \
  jevgames/strategies/*.py jevgames/tasks/*.py \
  sokoban_laya/*.py scripts/*.py
uv lock --check
```

Training-path changes should also pass the bounded smoke experiment:

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

## Documentation expectations

- update `docs/CONFIGURATION.md` for configuration changes;
- update `docs/PLUGINS.md` for contract changes;
- update schema versions for breaking artifact changes;
- distinguish implemented behavior from roadmap items;
- include measured benchmark evidence for performance claims.

## Pull requests

Keep pull requests focused. Describe the motivation, design boundary, tests,
and behavioral or benchmark impact. Do not commit generated checkpoints,
virtual environments, caches, secrets, or large generated datasets.

## License status

The repository does not currently declare an open-source license. Public
visibility alone does not grant permission to copy, modify, or redistribute the
code. A project license should be selected by the repository owner before
accepting substantive external contributions.
