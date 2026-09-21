# RLCD training semantics

Jev Games trains typed probability distributions with Reinforcement Learning
for Calibrated Decisions. The implementation follows the open Laya formulation
and does not claim to reproduce TypeSafe's unpublished Jev algorithm.

## Evidence

The task returns `CalibrationDecision` values containing an observation,
instruction, dynamic option set, and target distribution:

```python
CalibrationDecision(
    serialized_state="...",
    observation={"field": "value"},
    instruction="Choose the best action",
    options=(DecisionOption("a", "first"), DecisionOption("b", "second")),
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

## Reported distribution

For evidence item `i`, the model produces masked logits `z_i` over only the
options supplied for that state:

```text
p_i = softmax(mask(z_i))
```

Invalid padding never receives probability mass.

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

The model adapter scores each reported distribution against the target. Laya
uses a sum of log and spherical scores, with ranked probability score support
for ordinal questions:

```text
R(q, y) = log_score(q, y) + 0.5 × spherical_score(q, y)
```

A strictly proper scoring rule has its best expected reward when the reported
distribution equals the true outcome distribution. This is the defining
objective; no cross-entropy imitation term or game solve reward is mixed into
the built-in strategy.

## Group-relative baseline

For each evidence item, sampled rewards are normalized within its own group:

```text
A_g = (R_g - mean_g(R)) / (std_g(R) + 1e-6)
```

Items whose sampled rewards are identical receive zero advantage. Normalizing
per evidence item avoids comparing raw proper scores from unrelated option
sets.

## REINFORCE update

The Gaussian policy is centered on the live model logits. The sampled logits
are detached, while their log density is evaluated under the live mean:

```text
log π_z(z_sample) = -Σ_valid (z_sample - z)² / (2σ²)
L_RLCD = -mean(A × log π_z(z_sample))
```

Accelerate owns device placement, mixed precision, accumulation, synchronized
backward, and gradient clipping. AdamW performs the update.

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
