# Architecture

Jev Games separates model semantics, task environments, evidence datasets,
optimization strategy, and experiment configuration. The framework coordinates
them without importing a particular game or model into its core.

## Component boundaries

```text
ExperimentConfig
├── model.type ──────> DecisionModelAdapter
│                     encode, logits, masks, proper score, checkpoint
├── task.type ───────> DecisionTask
│                     observations, options, transitions
├── data.type ───────> EvidenceProvider
│                     dataset to calibration targets
├── training.type ───> TrainingStrategy
│                     sampling, optimization, metrics
└── benchmark ───────> lifecycle and held-out evaluation
```

The four registries are peers. A new model, task, evidence format, or training
algorithm can be selected independently in TOML.

### Model adapter

The adapter converts generic `DecisionOption` values into the model's native
input representation. It exposes option logits, the valid-option mask, target
distributions, a strictly proper calibration score, and native checkpoint
serialization.

The framework does not assume token generation. Laya, for example, is a
bidirectional encoder that scores option markers in one forward pass.

### Task adapter

The task owns domain state and transition semantics. It provides:

- state-dependent typed options;
- state transitions for model-only environment evaluation;
- canonical state identity and terminal conditions.

### Evidence provider

The provider reads a dataset and uses a compatible task to construct
`CalibrationDecision` rows. Targets may be one-hot observed outcomes or soft
distributions formed from repeated outcomes. The training strategy does not know how they were
produced.

### Training strategy

The strategy receives encoded evidence through the model adapter. The built-in
`rlcd` strategy owns Gaussian exploration, group-relative baselines, REINFORCE,
gradient accumulation, clipping, and device orchestration through Accelerate.

The strategy never calls the Sokoban environment and does not consume shaped
game reward. That boundary prevents solve reward from silently replacing the
calibration objective.

## Experiment lifecycle

```text
load TOML
  → construct model, task, evidence, and training strategy plugins
  → load calibration evidence through the data plugin
  → encode decisions with the model adapter
  → optimize with RLCD
  → save a native model checkpoint
  → evaluate held-out calibration evidence
  → run greedy held-out environment play
  → write schema-versioned report.json
```

Validation and test datasets are never passed to the strategy. Both are read
only after the checkpoint has been trained.

## Evidence and environment are different interfaces

For Sokoban, a solver path provides an observed successful push at a state. It
becomes a one-hot target distribution. A future evidence builder can instead
evaluate every legal push across repeated continuations and emit a soft target.

Environment play asks a different question: can the saved greedy policy solve
the full level without search fallback? It measures sequential competence, not
probability calibration.

## Dependency direction

`jevgames.engine` imports only contracts and evaluation configuration.
`jevgames.experiment` imports registries and the engine. Built-in adapters may
import their implementation packages, but the generic engine never imports
Laya or Sokoban.

Legacy task-specific SFT and environment-GRPO experiments remain in
`sokoban_laya`. They are compatibility and ablation tools, not dependencies of
the main experiment lifecycle.

## Current limits

- no optimizer resume or automatic best-checkpoint selection;
- no entry-point discovery for third-party plugin packages;
- one training strategy is currently registered;
- distributed optimizer preparation exists, but experiment artifact writing
  has only been validated with one process;
- Sokoban's current evidence is solver-derived one-hot data, not repeated
  empirical success probabilities for every legal action.
