# Framework selection

Jev Games delegates general training mechanics to established libraries while
retaining a small domain-specific layer for dynamic decisions and environment
semantics.

## Decision summary

| Concern | Selected framework | Rationale |
|---|---|---|
| Supervised expert warm-up | Transformers Trainer | Complete optimizer/dataloader/logging loop with custom loss support |
| Device and online optimization | Accelerate | Portable MPS/CUDA placement, accumulation, clipping, synchronization |
| Legal dynamic actions | TorchRL `MaskedCategorical` | One distribution API for sampling, log-probability, entropy, and mode |
| Online objective | Jev Games | Group-relative environment return is task/model agnostic but not token generation |
| Model implementation | Adapter plugin | Preserve native architecture and checkpoint format |
| Environment implementation | Task plugin | Preserve domain state, actions, rewards, and terminal rules |

Primary documentation:

- [Transformers Trainer](https://huggingface.co/docs/transformers/trainer)
- [Trainer customization](https://huggingface.co/docs/transformers/trainer_customize)
- [Accelerate on Apple MPS](https://huggingface.co/docs/accelerate/v0.12.0/en/usage_guides/mps)
- [TorchRL MaskedCategorical](https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.MaskedCategorical.html)
- [TorchRL PPO tutorial](https://docs.pytorch.org/rl/main/tutorials/coding_ppo.html)

## Transformers Trainer

Trainer is used only where its abstraction fits: supervised batches with a
scalar loss. Jev Games supplies a custom `compute_loss` that delegates to the
model adapter. Trainer retains ownership of dataloading, optimizer creation,
gradient accumulation, clipping, scheduling, and logs.

This boundary means a model adapter can define cross-entropy, proper scoring,
ranking, multi-task, or another differentiable expert objective without
reimplementing the loop.

## Accelerate

Online environment collection does not fit Trainer's dataset iteration model.
Accelerate therefore owns the lower-level optimization mechanics for grouped
rollouts. It also provides a path from local MPS development to CUDA process
orchestration.

The current collector is not process-sharded. Accelerate can synchronize the
optimizer, but launching multiple processes today would duplicate environment
collection. Distributed efficiency requires explicit state sharding first.

## TorchRL

Jev Games tasks may expose a different number of choices at every state.
Encoded batches are padded, and the model adapter supplies a boolean action
mask. TorchRL's `MaskedCategorical` guarantees that invalid padded choices have
zero probability and are excluded consistently from:

- stochastic rollout sampling;
- old and new action log-probabilities;
- entropy regularization;
- greedy benchmark selection.

TorchRL collectors and TensorDict environments were not adopted yet because
task states can be arbitrary Python objects and model observations can be
variable-length text. They remain a future scaling option after the generic
contracts stabilize.

## Why not TRL GRPOTrainer

TRL `GRPOTrainer` is designed around causal language models that generate
completion tokens. Its model contract, rollout, and reward pipeline assume
prompt/completion generation. Jev Games supports arbitrary decision models,
including bidirectional encoders that score a set of option markers in one
forward pass.

The built-in Laya model is such an encoder. Converting each decision into text
generation solely to satisfy TRL would change action semantics, latency,
calibration, and checkpoint compatibility.

Reference: [TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.27.1/en/grpo_trainer).

TRL becomes appropriate if a future adapter intentionally uses a supported
causal LM and token-generation environment. That adapter can coexist with the
current decision adapters rather than replacing the core contracts.

## Why not Unsloth

Unsloth provides strong optimizations for supported generative LLM families,
quantization, LoRA, and CUDA-oriented fine-tuning. The current model is a custom
ModernBERT decision architecture and the validated local target is Apple MPS.
Its Python installation guidance and optimized kernels do not provide a direct
drop-in path for this architecture.

Reference: [Unsloth installation matrix](https://docs.unsloth.ai/get-started/installing-%2B-updating/pip-install).

Unsloth may become useful for a future causal-LM adapter on supported hardware.
It is not a generic replacement for the Jev Games engine.

## Why not Stable-Baselines3

Stable-Baselines3 is optimized for Gym-like numeric observations and fixed
action spaces. Jev Games jointly encodes structured/text observations and
state-specific textual options. Mapping those options to a fixed global action
space would require a custom policy and masking layer and would discard much of
the benefit of the library's standard policies.

## Why not RLlib

RLlib can support custom models, action masking, and distributed workers. It is
a candidate when rollout fleets become the primary bottleneck. For the current
single-M4 environment it adds substantial process, serialization, and
configuration complexity while still requiring custom model and task code.

A future RLlib backend should implement the existing contracts rather than
changing dataset or model plugins.

## Why not Lightning

Lightning or Fabric could replace parts of Trainer/Accelerate, but would not
remove the need for a dynamic-action distribution, environment collector, or
model-specific adapter. The current stack already covers local MPS and a CUDA
scaling path with fewer overlapping abstractions.

## Framework ownership rule

Project code should be added only when at least one is true:

- it defines task semantics;
- it adapts a model's native interface;
- it implements the group-relative algorithm not offered for this model type;
- it coordinates framework phases and artifacts;
- it enforces a benchmark invariant.

Project code should not reimplement generic dataloading, optimizer stepping,
gradient clipping, masked categorical math, or device placement when the
selected frameworks already provide them.
