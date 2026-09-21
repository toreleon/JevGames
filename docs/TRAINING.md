# Training semantics

Jev Games uses a two-phase training strategy:

1. supervised expert warm-up establishes the action representation and a
   calibrated initial policy;
2. online group-relative optimization improves decisions using complete
   environment outcomes while replaying expert examples.

Both phases operate through model and task contracts. The engine has no
model-specific tokenizer logic and no task-specific reward logic.

## Phase 0: expert transformation

The task adapter loads source records and returns `ExpertDecision` objects:

```python
ExpertDecision(
    serialized_state="...",
    observation={"field": "value"},
    instruction="Choose the best action",
    options=(
        DecisionOption("a", "first action"),
        DecisionOption("b", "second action"),
    ),
    target_key="b",
    episode_id="episode-42",
    step=3,
)
```

The engine removes examples with fewer than two options. Forced decisions do
not contain policy information and are executed directly during rollout.

The model adapter converts each remaining example to its native encoded batch
representation.

## Phase 1: supervised warm-up

Transformers Trainer controls the standard optimization loop. Jev Games
subclasses only `compute_loss` and delegates the loss to the model adapter.

For the built-in Laya adapter:

```text
L_warmup = cross_entropy(target_action)
         + 0.25 × proper_scoring_loss(target_distribution)
```

Cross-entropy quickly teaches a new action schema. The proper-scoring term
discourages an arbitrarily sharp, poorly calibrated probability distribution.
Another model adapter may implement a different loss without changing Trainer
or the task.

Trainer owns:

- shuffling and dataloading;
- optimizer construction;
- gradient accumulation;
- gradient clipping;
- epoch control;
- training metrics and timing.

Setting `warmup.epochs = 0` skips this phase. Pure online training is supported
mechanically but generally unsuitable for sparse-reward tasks without an
already competent policy.

## Phase 2: grouped online collection

For every initial state, the engine creates `group_size` independent episodes.
At each state:

1. the task provides the currently legal options;
2. forced single-option transitions execute directly;
3. multi-option observations are batched through the model adapter;
4. TorchRL `MaskedCategorical` samples only legal actions;
5. the task applies the action and returns reward and terminal metadata;
6. the engine records the old action log-probability for the later update.

The task defines state identity. This is important for environments where
different raw states are behaviorally equivalent. The Sokoban plugin, for
example, canonicalizes all player positions in the same reachable region.

Episodes terminate on one of:

- task success;
- task-defined failure or deadlock;
- repeated canonical state;
- no legal action;
- the configured decision limit.

## Group-relative advantage

For a group of returns `R_1 ... R_G`, Jev Games computes:

```text
A_i = (R_i - mean(R)) / (std(R) + 1e-6)
```

Every trainable policy decision in episode `i` receives `A_i`. If all group
returns are equal, the standard deviation is effectively zero and the group is
discarded. This is deliberate: the group provides no relative preference.

## Clipped policy objective

The policy ratio for a sampled action is:

```text
r_t = exp(log π_new(a_t | s_t) - log π_old(a_t | s_t))
```

The online policy term is:

```text
L_policy = -mean(min(r_t A_t,
                     clip(r_t, 1-ε, 1+ε) A_t))
```

The complete update is:

```text
L_total = L_policy
        + expert_weight × L_expert_replay
        - entropy_weight × H(π)
```

Expert replay reduces catastrophic forgetting and keeps the policy anchored to
known valid decisions. Entropy encourages exploration. Accelerate controls
backward, accumulation, clipping, and optimizer synchronization.

## Dynamic action masking

The action mask is part of the model adapter contract. Invalid padded options
must be false in the mask. TorchRL renormalizes the distribution over valid
options and uses the same mask for sampling, log-probability, entropy, and
greedy evaluation.

Tasks should avoid presenting semantically duplicate actions. Duplicate
options split probability mass and create ambiguous expert targets.

## Reproducibility

The experiment seed is passed to Trainer and PyTorch sampling. Dataset
generators should expose and record their own seeds. Determinism is bounded by
the selected PyTorch backend; MPS and CUDA kernels may not be bit-identical
across versions or devices.

For credible comparisons:

- pin `uv.lock`;
- retain the full TOML configuration;
- retain dataset manifests and content hashes;
- evaluate the same checkpoint on the same held-out split;
- report hardware, framework versions, and seed;
- run multiple seeds before claiming an algorithmic improvement.

## Backbone freezing

Freezing a pretrained backbone is recommended for local MPS development:

- substantially lower optimizer memory;
- faster backward passes;
- reduced catastrophic forgetting;
- practical iteration on a 48 GB unified-memory machine.

Unfreezing can improve representation adaptation when the task distribution is
far from pretraining, but it changes resource requirements and should be
validated separately.

## Current limitations

The generic engine currently does not provide:

- resumable optimizer/scheduler state;
- automatic best-checkpoint selection;
- generalized advantage estimation or a learned value function;
- rollout replay buffers;
- asynchronous environment workers;
- per-process dataset sharding for distributed online collection;
- KL regularization against a frozen reference policy.

These omissions are explicit. Long production runs should not be described as
fault-tolerant until resume and best-checkpoint support are implemented.
