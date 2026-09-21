# Command-line interface

The public console command is `jev-games`.

## `jev-games plugins`

List registered model and task plugin names:

```bash
uv run jev-games plugins
```

Output is JSON and can be consumed by automation.

## `jev-games run CONFIG`

Run an end-to-end experiment:

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

The command blocks until warm-up, online training, checkpoint export, and all
configured benchmarks finish. Final output is a compact JSON summary; the full
report is written to `<output_dir>/report.json`.

Exit behavior:

- returns zero on success;
- propagates configuration, plugin, dataset, model, and runtime exceptions;
- does not currently recover or resume a partial run.

## Script utilities

The repository includes task and research utilities that are intentionally not
part of the generic CLI.

### Generate Sokoban curriculum

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot \
  --train 2000 --validation 200 --test 200
```

### Estimate training time

```bash
uv run python scripts/estimate_training_time.py \
  models/laya-sokoban-hf-pilot100/grpo_stats.json \
  --benchmark-levels 100 \
  --output data/pilot/framework_training_estimate.json
```

### Evaluate one Sokoban level

```bash
uv run python scripts/evaluate_macro_model.py \
  runs/example/checkpoint levels/microban_3.xsb --device mps
```

## Compatibility CLI

The `sokoban-laya` command exposes earlier task-specific experiments and
ablation paths. It is retained for reproducibility, not recommended as the
primary framework interface.

```bash
uv run sokoban-laya --help
```

New model/task integrations should use `jev-games` contracts and TOML
experiments.
