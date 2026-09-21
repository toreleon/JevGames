# Design sources and interpretation

Jev Games distinguishes published concepts, open implementation details, and
local design decisions.

## Primary sources

- [Laya engineering write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
- [Laya repository](https://github.com/NandhaKishorM/laya)
- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [TypeSafe introduction to System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

The Laya sources publish an executable architecture and fine-tuning notebook.
TypeSafe publishes the RLCD name and calibration objective but not a complete
reproducible training algorithm.

## Adopted as framework contracts

- non-autoregressive probability distributions over runtime-defined options;
- typed `choice`, ordinal `score`, and false/true `noul` questions;
- composite log, spherical, and ordinal ranked probability scores;
- zero-mean Gaussian logit exploration with group-relative REINFORCE;
- normalized-entropy confidence and selective-accuracy reporting;
- post-training temperature fitting on a dedicated held-out split;
- invalid-action masking as a feasibility constraint, separated from episodic
  exploration policy;
- model-native checkpoint persistence.

These concepts live outside the Laya adapter so another encoder, graph model,
vision model, or multimodal decision model can implement the same contracts.

## Deliberate boundaries

The current default strategy uses pure RLCD. The public article describes pure
policy gradient, while the current open fine-tuning notebook adds a
cross-entropy guidance term. A future hybrid strategy should have a separate
name and explicit coefficient.

The article also describes a learned act/escalate head and TD(λ) for multi-turn
trajectories. Jev Games currently reports confidence/coverage curves and
preserves evidence IDs and steps, but it does not yet train a generic gate or
TD targets. Those require explicit policy-cost and trajectory contracts rather
than assumptions inside the Laya adapter.

The `qwen_decision` adapter demonstrates that RLCD is not tied to Laya. It uses
a causal-safe candidate scorer; it does not claim Qwen's ordinary next-token
probabilities are calibrated action probabilities.

Sokoban currently supplies one-hot successful-solver actions. That evidence
does not equal empirical action success probability. An outcome evidence
provider remains necessary before claiming calibrated probabilities of solving
a level.
