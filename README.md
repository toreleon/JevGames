# Jev Games

Jev Games is a pluggable framework for applying calibrated, typed decision
models to interactive tasks. It trains distributions over state-dependent
choices rather than generating action text.

The primary training strategy is RLCD: Reinforcement Learning for Calibrated
Decisions. A task supplies observed decision outcomes, a model reports a
probability distribution over the available options, and a strictly proper
scoring rule supplies the reward. Group-relative REINFORCE updates make honest
probabilities optimal in expectation.

Sokoban is the first task integration and Laya is the first model integration.
Neither is embedded in the framework core.

## Architecture

```text
                         TOML experiment
                                │
             ┌─────────────┬────┴────────┬──────────────┐
             │             │             │              │
        model.type     task.type     data.type    training.type
        Laya adapter  Sokoban env   solver data       RLCD
             │             │             │              │
             └─────────────┴──────┬──────┴──────────────┘
                                │
                  typed probability distributions
                                │
                 strictly proper scoring reward
                                │
          Gaussian exploration + group-relative REINFORCE
                                │
             calibration benchmark + environment benchmark
```

The four plugin contracts are independent:

- a model adapter encodes typed options, produces logits, defines its proper
  scoring rule, and preserves its native checkpoint format;
- a task adapter exposes typed environment state and transitions;
- an evidence provider converts a dataset into one-hot or soft calibration
  targets for a compatible task;
- a training strategy owns sampling, optimization, device orchestration, and
  training statistics.

Environment rewards do not enter the built-in RLCD objective. An evidence
provider may use a solver or repeated environment outcomes to construct
one-hot or soft target distributions before training.

## Quick start

Requirements:

- Python 3.11 or 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- Apple Silicon with PyTorch MPS, NVIDIA CUDA, or CPU for small verification.

Install the Laya and training integrations:

```bash
uv sync --extra laya --extra train --python 3.12
```

Inspect the plugin registry:

```bash
uv run jev-games plugins
```

```json
{
  "evidence": ["sokoban_solver"],
  "models": ["laya"],
  "strategies": ["rlcd"],
  "tasks": ["sokoban_push"]
}
```

Generate the small held-out curriculum and run the bounded experiment:

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot100 --train 100 --validation 20 --test 20 --seed 20260921
uv run jev-games run configs/sokoban_smoke.toml
```

The run performs RLCD optimization, exports a native Laya checkpoint, measures
held-out calibration, plays the held-out environments greedily, and writes:

```text
runs/laya-sokoban-smoke/
├── checkpoint/
└── report.json
```

Generate and train the 2,000-level pilot:

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot --train 2000 --validation 200 --test 200 --seed 20260921
uv run jev-games run configs/sokoban_pilot.toml
```

## Configuration example

```toml
[experiment]
name = "laya-sokoban-rlcd"
output_dir = "runs/laya-sokoban-rlcd"
seed = 20260921

[model]
type = "laya"
source = "convaiinnovations/laya"
device = "auto"
freeze_backbone = true

[task]
type = "sokoban_push"

[training]
type = "rlcd"
epochs = 4
batch_size = 16
gradient_accumulation = 2
group_size = 4
sigma_start = 0.5
sigma_end = 0.2
learning_rate = 0.0003
mixed_precision = "no"

[data]
type = "sokoban_solver"
train = "data/pilot/train.jsonl"
validation = "data/pilot/validation.jsonl"
test = "data/pilot/test.jsonl"
manifest = "data/pilot/manifest.json"

[benchmark]
max_decisions = 32
batch_size = 32
calibration_bins = 15
```

The old `[warmup]` and `[online]` sections are rejected. This prevents a stale
SFT-plus-environment-GRPO experiment from silently running under the RLCD
architecture.

## What is measured

Every held-out split reports two independent views:

- calibration: accuracy, soft accuracy, negative log-likelihood, multiclass
  Brier score, expected calibration error, mean confidence, mean proper score,
  and reliability bins;
- environment: solve rate, terminal outcomes, push decisions, and primitive
  movement cost.

A higher solve rate does not prove calibrated probabilities, and low expected
calibration error does not prove long-horizon planning. Both contracts must be
reported.

## RLCD implementation scope

TypeSafe publicly describes RLCD as Reinforcement Learning for Calibrated
Decisions, but has not published a complete reproducible algorithm. Jev Games
implements the open Laya formulation:

1. add zero-mean Gaussian exploration to option logits;
2. convert sampled logits to typed probability distributions;
3. reward each distribution with a strictly proper scoring rule;
4. normalize rewards within the sample group;
5. update the distribution mean with REINFORCE.

This is an open implementation inspired by the stated objective; it is not a
claim of reproducing TypeSafe's proprietary Jev training stack.

## Repository layout

```text
jevgames/
  contracts.py             Model, task, evidence, and strategy interfaces
  registry.py              Explicit plugin registry
  experiment.py            End-to-end orchestration
  engine.py                Evidence encoding and evaluation
  evidence/                Dataset/evidence provider plugins
  models/laya.py           Laya adapter
  tasks/sokoban.py         Sokoban environment adapter
  strategies/rlcd.py       RLCD optimization strategy
sokoban_laya/              Sokoban and legacy experiment implementation
configs/                    Reproducible TOML experiments
scripts/                    Dataset and analysis utilities
tests/                      Contract and domain regressions
docs/                       Detailed documentation
```

Legacy SFT/GRPO tools remain under `sokoban_laya` for comparison experiments.
They are not part of the default Jev Games experiment lifecycle.

## Documentation

- [Documentation index](docs/INDEX.md)
- [Getting started](docs/GETTING_STARTED.md)
- [Architecture](docs/ARCHITECTURE.md)
- [RLCD training semantics](docs/TRAINING.md)
- [Configuration](docs/CONFIGURATION.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Plugin API](docs/PLUGINS.md)
- [Artifacts](docs/ARTIFACTS.md)
- [Operations and scaling](docs/OPERATIONS.md)
- [Sokoban case study](docs/SOKOBAN.md)
- [Framework selection](docs/FRAMEWORKS.md)
- [Development](docs/DEVELOPMENT.md)

## Verification

```bash
uv run --extra laya --extra train python -m unittest discover -s tests -v
uv run python -m py_compile jevgames/*.py jevgames/*/*.py sokoban_laya/*.py
uv lock --check
```

## License status

No open-source license has been selected yet. Public availability does not by
itself grant permission to copy, modify, or redistribute this repository.
