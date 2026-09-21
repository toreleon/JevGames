# Jev Games

Jev Games is a pluggable framework for training and benchmarking models that
choose from a dynamic set of decisions while interacting with an environment.
The training engine is independent of any specific model, dataset, or game.

The framework combines:

- Hugging Face Transformers Trainer for supervised expert warm-up;
- Hugging Face Accelerate for optimization and device orchestration;
- TorchRL masked categorical distributions for legal-action sampling;
- model adapters for tokenization, logits, losses, and checkpoint formats;
- task adapters for datasets, environment transitions, rewards, and terminal
  conditions;
- config-driven experiments with model-only validation and test benchmarks.

The first built-in integration uses Laya as the decision model and Sokoban
push macros as the task. Both are plugins rather than framework dependencies.

## Architecture at a glance

```text
                         TOML experiment
                                │
                    ┌───────────┴───────────┐
                    │                       │
             Model registry          Task registry
                    │                       │
        DecisionModelAdapter          DecisionTask
                    │                       │
                    └───────────┬───────────┘
                                │
                  Generic Jev Games engine
             ┌──────────────────┼──────────────────┐
             │                  │                  │
       Expert warm-up       Online GRPO        Benchmark
       HF Trainer           Accelerate         Model-only
                            + TorchRL
```

The generic engine never imports Laya or Sokoban. A new model and a new task
can be combined by selecting their registry names in configuration.

## Quick start

Requirements:

- Python 3.11 or 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- macOS with Apple Silicon for MPS, or a supported CUDA environment.

Install the current built-ins and training stack:

```bash
uv sync --extra laya --extra train --python 3.12
```

Inspect available plugins:

```bash
uv run jev-games plugins
```

Expected output:

```json
{
  "models": ["laya"],
  "tasks": ["sokoban_push"]
}
```

Run the bounded smoke experiment:

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

The experiment performs expert warm-up, online group-relative optimization,
checkpoint export, validation, test evaluation, and report generation. Its
primary artifact is:

```text
runs/laya-sokoban-smoke/report.json
```

Run the full 2,000-level pilot configuration:

```bash
uv run jev-games run configs/sokoban_pilot.toml
```

On the measured Apple M4 Pro 48 GB system, this configuration is estimated at
approximately 4.8 hours, with ±25% uncertainty.

## Configuration example

```toml
[experiment]
name = "laya-sokoban-pilot-2000"
output_dir = "runs/laya-sokoban-pilot-2000"
seed = 20260921

[model]
type = "laya"
source = "models/laya-sokoban-smoke"
device = "mps"
freeze_backbone = true

[task]
type = "sokoban_push"
solve_reward = 50.0
goal_reward = 4.0
deadlock_penalty = 15.0

[data]
train = "data/pilot/train.jsonl"
validation = "data/pilot/validation.jsonl"
test = "data/pilot/test.jsonl"
manifest = "data/pilot/manifest.json"

[warmup]
epochs = 6
batch_size = 16
gradient_accumulation = 2
learning_rate = 0.0003

[online]
iterations = 2
group_size = 4
max_decisions = 32
max_instances = 2000
batch_size = 16
rollout_instance_batch_size = 16
rollout_inference_batch_size = 32

[benchmark]
max_decisions = 32
batch_size = 32
```

## Documentation

- [Documentation index](docs/INDEX.md)
- [Getting started](docs/GETTING_STARTED.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [Training semantics](docs/TRAINING.md)
- [Benchmarking protocol](docs/BENCHMARKING.md)
- [Plugin API](docs/PLUGINS.md)
- [Artifacts and schemas](docs/ARTIFACTS.md)
- [Operations and scaling](docs/OPERATIONS.md)
- [CLI reference](docs/CLI.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Framework selection research](docs/FRAMEWORKS.md)
- [Sokoban case study](docs/SOKOBAN.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Repository layout

```text
jevgames/                 Generic framework package
  contracts.py            Model and task interfaces
  registry.py             Plugin registration and construction
  config.py               TOML configuration schema
  engine.py               Warm-up, online GRPO, and evaluation
  experiment.py           End-to-end orchestration
  models/                  Built-in model adapters
  tasks/                   Built-in task adapters
sokoban_laya/             Sokoban/Laya implementation and legacy tools
configs/                   Reproducible experiment configurations
scripts/                   Dataset, benchmark, and compatibility utilities
tests/                     Framework and integration tests
docs/                      Design and operating documentation
```

## Development verification

```bash
uv run --extra laya --extra train python -m unittest discover -s tests -v
uv lock --check
```

The current suite contains 20 passing tests. The smoke experiment has also
been executed end-to-end on Apple M4 Pro using PyTorch MPS.

## Current scope

Jev Games currently supports dynamic discrete decisions, expert warm-up,
group-relative online optimization, model-only evaluation, and local MPS/CUDA
execution. The following are explicit future work rather than current claims:

- resumable optimizer checkpoints;
- automatic best-checkpoint selection;
- process-sharded rollout collection for multi-GPU jobs;
- Python entry-point discovery for third-party plugin packages;
- distributed environment worker fleets.

The legacy `sokoban-laya` command remains available for reproducing earlier
experiments, but new work should use `jev-games` and the generic contracts.

## License status

No open-source license has been selected yet. Public availability does not by
itself grant permission to copy, modify, or redistribute the code. The
repository owner should add an explicit license before inviting substantive
external reuse or contributions.
