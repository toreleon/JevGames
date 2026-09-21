# RLCD training semantics

Jev Games trains typed probability distributions with Reinforcement Learning
for Calibrated Decisions. The implementation follows the open Laya formulation
and does not claim to reproduce TypeSafe's unpublished Jev algorithm.

## Evidence

The evidence provider returns `CalibrationDecision` values containing an
observation, typed question, dynamic option set, and target distribution:

```python
CalibrationDecision(
    serialized_state="...",
    observation={"field": "value"},
    question=DecisionQuestion(
        key="route",
        kind=DecisionKind.CHOICE,
        instruction="Choose the best action",
        options=(DecisionOption("a", "first"), DecisionOption("b", "second")),
    ),
    target_probabilities=(0.75, 0.25),
    evidence_id="episode-42",
    step=3,
    provenance="environment_outcomes",
)
```

Targets must be non-negative, match the option count, and sum to one. Forced
single-option decisions are removed because they provide no learning signal.

One-hot targets represent individual observed outcomes. Soft targets can
represent repeated outcomes or a teacher distribution. They are first-class
inputs rather than a model-specific extension.

## Typed questions

The core supports the three Laya/System-1 primitives without depending on the
Laya implementation:

- `choice`: unordered categorical options;
- `score`: ordered rubric levels, which receive ranked probability score;
- `noul`: options fixed as `false, true`, reporting `P(true)` directly.

Evidence providers can emit several question keys for one serialized state.
Adapters may collate those questions into one hardware batch. `evidence_id` and
`step` preserve trajectory grouping for future TD(λ) strategies; the current
RLCD strategy treats each labeled decision independently.

## Reported distribution

For evidence item `i`, the model produces masked logits `z_i` over only the
options supplied for that state:

```text
p_i = softmax(mask(z_i))
```

Invalid padding never receives probability mass.

RLCD consumes `training_logits`, which are raw with respect to post-training
temperature transforms. Held-out inference consumes calibrated `logits`.

## Exploration

RLCD draws `G` zero-mean Gaussian perturbations for every evidence item:

```text
ε_g ~ N(0, σ²I)
ε_g ← ε_g - mean_valid_options(ε_g)
q_g = softmax(mask(stop_gradient(z) + ε_g))
```

`sigma_start` and `sigma_end` define a linear epoch schedule. Centering the
noise prevents a meaningless common logit shift from consuming exploration.

## Proper scoring reward

The framework core scores each reported distribution with log and spherical
scores, plus ranked probability score for ordinal `score` questions:

```text
R(q, y) = log_score(q, y)
        + 0.5 × spherical_score(q, y)
        - 1.0 × ordinal_rps(q, y)
```

A strictly proper scoring rule has its best expected reward when the reported
distribution equals the true outcome distribution. This is the defining
objective; no cross-entropy imitation term or game solve reward is mixed into
the built-in strategy.

The logarithmic term is not clipped at an arbitrary reward floor. It clamps
only to the smallest positive float required to evaluate `log(0)` safely;
clipping the score itself can break strict propriety for rare outcomes.

## Group-relative baseline

For each evidence item, sampled rewards are normalized within its own group:

```text
A_g = (R_g - mean_g(R)) / (std_g(R) + 1e-6)
```

Items whose sampled rewards are identical receive zero advantage. Normalizing
per evidence item avoids comparing raw proper scores from unrelated option
sets.

**Known limitation:** division by the within-example reward standard deviation
also changes the relative weight of different observed labels. The update is
not guaranteed to preserve the proper reward's optimum. In a constant-input
experiment with 30% true outcomes, the current normalized update predicts
approximately 0.008% true, while direct optimization predicts 30%. See the
[reproducible normalization diagnostic](ADVANTAGE_NORMALIZATION_DIAGNOSTIC.md).
The `direct_proper_score` strategy uses the same optimizer infrastructure and
directly differentiates the composite score. The existing `rlcd` strategy is
retained unchanged so the comparison remains reproducible.

## REINFORCE update

The Gaussian policy is centered on the live model logits. The sampled logits
are detached, while their log density is evaluated under the live mean:

```text
log π_z(z_sample) = -Σ_valid (z_sample - z)² / (2σ²)
L_RLCD = -mean(A × log π_z(z_sample))
```

Accelerate owns device placement, mixed precision, accumulation, synchronized
backward, and gradient clipping. AdamW performs the update.

## State exploration and online evidence

RLCD logit noise explores probability reports for a known decision. It does
not visit a new environment state. When `[collection]` is enabled, Jev Games
adds a separate loop:

```text
current calibrated policy
  → enumerate executable actions
  → ask one noul outcome question per action
  → sample with temperature + epsilon
  → execute the selected action
  → label selected decisions with terminal episode success
  → replay offline and collected evidence through RLCD
```

The outcome event is explicitly conditional on the collector's current
continuation policy and decision horizon. A failed episode is evidence about
that policy, not proof that the first action can never participate in a
solution. Each update replays offline evidence plus the latest collection;
older on-policy outcomes are not mixed across policy versions.

Legal-action masking removes impossible transitions. It does not remove legal
deadlocks, cycles, or strategically weak pushes. Epsilon controls how often the
collector tries low-scored legal actions.

## Post-training calibration

Training changes the logits, so temperatures fitted for the base checkpoint
are stale. When `data.calibration` is configured, the lifecycle asks the model
adapter to fit native calibration parameters on that dedicated split before it
saves the checkpoint. Laya fits scalar temperatures by decision type and
option-count bucket with held-out negative log-likelihood.

Validation and test remain untouched by both weight training and temperature
fitting. A calibration split is part of model construction, not a benchmark.

## What the probability means

Calibration is only meaningful relative to the target event. In the current
Sokoban integration, the model reports a distribution over which legal push
matches an observed successful solver trajectory. That is not automatically a
probability that a push will solve the entire level.

To train the latter quantity, the evidence builder must evaluate each legal
push with repeated controlled continuations and define the target distribution
from those outcomes. Solver timeout must remain unknown evidence rather than be
recorded as failure.

## Backbone adaptation

`model.freeze_backbone = true` trains only the Laya decision layers. This is the
safe MPS baseline and uses substantially less memory. Set it to `false` to
adapt the encoder; the Laya adapter enables gradient checkpointing by default.
Full-backbone runs need a lower learning rate and hardware-specific validation.

## Reproducibility

Retain the TOML, `uv.lock`, data manifest, checkpoint metadata, hardware
description, and random seed. Compare checkpoints on identical held-out splits
and report multiple seeds before claiming an algorithmic improvement.

## Relationship to SFT and environment GRPO

SFT can be a separate initialization strategy and episode-reward GRPO can be an
ablation, but neither is part of the default RLCD lifecycle. The old
implementations remain under `sokoban_laya` so comparisons can be reproduced
without changing what `training.type = "rlcd"` means.

The current strategy intentionally keeps the decision loss pure RLCD. The
latest public Laya fine-tuning notebook also includes a cross-entropy guidance
term, while the accompanying article describes pure policy gradient. Jev Games
keeps any hybrid objective explicit rather than silently folding it into RLCD.
