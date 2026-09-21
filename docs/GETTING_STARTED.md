# Getting started

This guide installs Jev Games, verifies the accelerator, runs a complete smoke
experiment, and explains the generated artifacts.

## 1. Prerequisites

Supported project Python versions are 3.11 and 3.12. The repository is managed
with `uv`; do not install training dependencies into the system Python.

Verify `uv`:

```bash
uv --version
```

On Apple Silicon, use a current macOS release and a PyTorch build with the MPS
backend. CUDA users should use a PyTorch wheel appropriate for their driver and
GPU.

## 2. Install

From the repository root:

```bash
uv sync --extra laya --extra train --python 3.12
```

The extras have separate responsibilities:

| Extra | Purpose |
|---|---|
| `laya` | Built-in Laya adapter, PyTorch, Transformers, Safetensors, Hub client |
| `train` | Accelerate and TorchRL framework integrations |

The environment is written to `.venv` and pinned by `uv.lock`.

## 3. Verify hardware

Apple Silicon:

```bash
uv run python - <<'PY'
import torch
assert torch.backends.mps.is_available()
print(torch.__version__)
print(torch.backends.mps.get_name())
PY
```

CUDA:

```bash
uv run python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__)
print(torch.cuda.get_device_name())
PY
```

Jev Games does not silently reinterpret an unavailable requested accelerator.
The model adapter should fail early with a useful error.

## 4. Verify plugins

```bash
uv run jev-games plugins
```

The current built-ins are:

- model: `laya`;
- task: `sokoban_push`.

Plugin names are configuration identifiers. They are independent: adding a
second model does not require a second Sokoban implementation, and adding a
second task does not require modifying Laya.

## 5. Run tests

```bash
uv run --extra laya --extra train python -m unittest discover -s tests -v
uv lock --check
```

Tests cover environment rules, solver replay, procedural generation, macro
execution, group-relative advantages, plugin discovery, configuration parsing,
and benchmark aggregation.

## 6. Run the smoke experiment

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

The run performs these phases:

1. construct the configured model and task plugins;
2. load and transform expert decisions;
3. warm up the policy with Transformers Trainer;
4. collect online environment groups;
5. update with Accelerate and TorchRL;
6. save the model-native checkpoint;
7. evaluate validation and test states without search fallback;
8. write a machine-readable experiment report.

Expected output directory:

```text
runs/laya-sokoban-smoke/
├── checkpoint/
│   ├── encoder/config.json
│   ├── model.safetensors
│   ├── rl_agent_config.json
│   ├── tokenizer/
│   └── training_stats.json
└── report.json
```

The smoke dataset is intentionally tiny. A successful run validates mechanics,
not task generalization.

## 7. Generate the pilot dataset

The current Sokoban plugin includes a reverse-play generator with known
solutions:

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot \
  --train 2000 \
  --validation 200 \
  --test 200 \
  --seed 20260921
```

It records seeds and layout hashes and rejects layout overlap across splits.
The generated pilot contains approximately 20,000 macro decisions.

## 8. Run the pilot

```bash
uv run jev-games run configs/sokoban_pilot.toml
```

Measured planning estimate on Apple M4 Pro 48 GB:

- six expert warm-up epochs: about 1.9 hours;
- 16,000 grouped online episodes: about 2.9 hours;
- total: about 4.8 hours, with ±25% uncertainty.

See [Operations and scaling](OPERATIONS.md) before launching long runs.
