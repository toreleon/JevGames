# Command-line interface

## List plugins

```bash
uv run jev-games plugins
```

This prints registered model, task, evidence, collector, and strategy names.

## Run an experiment

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

The command blocks until RLCD training, configured collection/update cycles,
post-training calibration fitting, checkpoint export, and evaluation complete. The
final line is a compact JSON summary; the full result is `report.json`.

## Generate Sokoban evidence

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot \
  --train 2000 --calibration 200 --validation 200 --test 200 \
  --seed 20260921
```

## Legacy compatibility CLI

`sokoban-laya` retains solver, play, original notebook RLCD, SFT/GRPO ablation,
and task-specific benchmark commands:

```bash
uv run sokoban-laya --help
```

New reproducible work should use `jev-games` TOML experiments. Legacy commands
must be named explicitly in reports because their algorithms and report schemas
differ.
