# Getting started

## Install

Use a project-local Python 3.12 environment:

```bash
uv sync --extra laya --extra train --python 3.12
```

Extras:

| Extra | Purpose |
|---|---|
| `laya` | Laya, PyTorch, Transformers, Safetensors, Hub client |
| `train` | Accelerate and TorchRL compatibility dependencies |

Do not install training packages into the system Python.

## Verify the runtime

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print("MPS", torch.backends.mps.is_available())
print("CUDA", torch.cuda.is_available())
PY
```

List plugins and run tests:

```bash
uv run jev-games plugins
uv run python -m unittest discover -s tests -v
```

## Generate held-out data

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot100 \
  --train 100 --validation 20 --test 20 \
  --seed 20260921
```

The manifest records environment seeds and split hashes.

## Run the RLCD smoke experiment

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

The config downloads the public Laya checkpoint if needed. Its evidence plugin
converts solver trajectories into push decisions, trains one RLCD epoch, saves a native
checkpoint, and runs calibration plus environment benchmarks.

The smoke run verifies mechanics. Its tiny training file cannot establish
generalization.

## Generate the pilot

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot \
  --train 2000 --validation 200 --test 200 \
  --seed 20260921
```

Review `configs/sokoban_pilot.toml`, hardware memory, and smoke metrics before
starting the longer run.

## Read the report

`report.json` records schema version 2, full parsed config, plugin identities,
RLCD epoch statistics, checkpoint path, and separate calibration/environment
metrics for validation and test.
