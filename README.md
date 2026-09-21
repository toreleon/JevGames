# Jev Games

Jev Games is a pluggable framework for applying calibrated, typed decision
models to interactive tasks. It trains probability distributions for typed
questions rather than generating action text.

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
          ┌──────────┬─────────┬──┴───────┬──────────┬───────────┐
          │          │         │          │          │           │
      model.type task.type data.type collection.type training.type
      Laya/Qwen  Sokoban   solver    episodic         RLCD
          │          │         │      outcomes        │
          └──────────┴─────────┴──────┬───────────────┘
                                │
                  typed probability distributions
                                │
                 strictly proper scoring reward
                                │
          Gaussian exploration + group-relative REINFORCE
                                │
             calibration benchmark + environment benchmark
```

The five plugin contracts are independent:

- a model adapter encodes typed questions, produces logits and masks, fits its
  native calibration parameters, and preserves its checkpoint format;
- a task adapter exposes typed environment state and transitions;
- an evidence provider converts a dataset into one-hot or soft calibration
  targets for a compatible task;
- an environment collector samples legal actions, discovers new states, and
  turns terminal outcomes into additional calibration evidence;
- a training strategy owns sampling, optimization, device orchestration, and
  training statistics.

The core represents three decision primitives independently of any model:
`choice` for categorical routing, `score` for ordered rubrics, and `noul` for a
calibrated false/true probability. The composite proper-scoring reward also
lives in the framework core rather than the Laya adapter.

Environment rewards do not replace the RLCD objective. Episodic collection
labels selected per-action `noul` questions with terminal outcomes, adds those
rows to replay evidence, and runs another RLCD update.

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
  "collectors": ["episodic_outcomes"],
  "evidence": ["sokoban_solver"],
  "models": ["laya", "qwen_decision"],
  "strategies": ["rlcd"],
  "tasks": ["sokoban_push"]
}
```

Generate the small held-out curriculum and run the bounded experiment:

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot100 --train 100 --calibration 20 \
  --validation 20 --test 20 --seed 20260921
uv run jev-games run configs/sokoban_smoke.toml
```

The run performs RLCD optimization, fits post-training temperature parameters
on a dedicated calibration split, exports a native Laya checkpoint, measures
held-out validation/test calibration, and plays those environments greedily.

```text
runs/laya-sokoban-smoke/
├── checkpoint/
└── report.json
```

Generate and train the 2,000-level pilot:

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot --train 2000 --calibration 200 \
  --validation 200 --test 200 --seed 20260921
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
update_epochs = 1
batch_size = 16
gradient_accumulation = 2
group_size = 4
sigma_start = 0.5
sigma_end = 0.2
learning_rate = 0.0003
mixed_precision = "no"

[collection]
type = "episodic_outcomes"
iterations = 2
episodes_per_instance = 4
max_instances = 2000
max_decisions = 32
sampling_temperature = 1.0
epsilon = 0.1

[data]
type = "sokoban_solver"
train = "data/pilot/train.jsonl"
calibration = "data/pilot/calibration.jsonl"
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

Use Qwen3-0.6B through the causal decision adapter:

```bash
uv sync --extra qwen --extra train --python 3.12
uv run jev-games run configs/sokoban_qwen_smoke.toml
```

For memory-efficient backbone adaptation on a CUDA GPU, use the LoRA pilot:

```bash
uv run jev-games run configs/sokoban_qwen_lora_pilot_v2.toml
```

That profile keeps the Qwen3-0.6B base in BF16, retains the complete aligned
Sokoban outcome prompt within a measured 256-token budget, trains rank-8 PEFT adapters
and the FP32 decision head at separate learning rates, enables activation
checkpointing, and uses gradient accumulation for a bounded physical batch.
It trains the same per-action `noul` primitive used for environment control:
expert successors are positive, proven static deadlocks are negative, and
failed exploratory prefixes remain unknown unless the final action proves a
terminal failure. The earlier `sokoban_qwen_lora_pilot.toml` remains available
as the one-hot `choice` ablation.

The adapter scores every candidate from the final token of a sequence that has
already seen the complete state and option set. It does not interpret ordinary
next-token generation probabilities as action probabilities.

## What is measured

Every held-out split reports two independent views:

- calibration: accuracy, soft accuracy, negative log-likelihood, Brier score,
  ECE, top probability, normalized-entropy confidence, ordinal MAE, selective
  accuracy, proper score, and reliability bins;
- outcome calibration: the same probability metrics on fresh terminal-outcome
  episodes for per-action `noul` questions;
- environment: solve rate, terminal outcomes, push decisions, and primitive
  movement cost.

A higher solve rate does not prove calibrated probabilities, and low expected
calibration error does not prove long-horizon planning. Both contracts must be
reported.

Legal-action masks enforce environment rules. Sokoban keeps reachable pushes
that lead to deadlocks or poor positions, so the mask does not choose for the
model. Temperature and epsilon in the episodic collector control state-space
exploration; RLCD sigma controls distribution exploration on observed states.

## RLCD implementation scope

TypeSafe publicly describes RLCD as Reinforcement Learning for Calibrated
Decisions, but has not published a complete reproducible algorithm. Jev Games
implements the open Laya formulation:

1. add zero-mean Gaussian exploration to option logits;
2. convert sampled logits to typed probability distributions;
3. reward each distribution with the core composite strictly proper score;
4. normalize rewards within the sample group;
5. update the distribution mean with REINFORCE.

This is an open implementation inspired by the stated objective; it is not a
claim of reproducing TypeSafe's proprietary Jev training stack.

The typed primitives, composite scoring rule, Gaussian policy-gradient update,
normalized-entropy confidence, and post-training temperature fitting are
grounded in the creator's [Laya engineering write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
and [open repository](https://github.com/NandhaKishorM/laya).

## Repository layout

```text
jevgames/
  contracts.py             Model, task, evidence, and strategy interfaces
  registry.py              Explicit plugin registry
  experiment.py            End-to-end orchestration
  engine.py                Evidence encoding and evaluation
  scoring.py               Model-independent proper scoring rules
  collectors/              Environment evidence collection plugins
  evidence/                Dataset/evidence provider plugins
  models/laya.py           Laya adapter
  models/qwen.py           Causal Qwen decision adapter
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
- [Design sources and interpretation](docs/DESIGN_SOURCES.md)
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
