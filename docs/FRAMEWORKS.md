# Framework selection

Jev Games uses established libraries for general optimization mechanics while
keeping RLCD's decision-specific sampling explicit.

| Concern | Selected component |
|---|---|
| Tensor/model runtime | PyTorch |
| Device placement, mixed precision, accumulation, distributed preparation | Hugging Face Accelerate |
| Laya tokenizer and encoder implementation | Hugging Face Transformers through Laya |
| Qwen causal backbone | Hugging Face Transformers `AutoModel` |
| Typed decisions and composite proper scores | Jev Games core |
| RLCD sampling and score-function objective | Jev Games strategy plugin |
| Task state and transition semantics | Task plugin |
| Dataset parsing and target provenance | Evidence plugin |
| Native checkpoint compatibility | Model adapter |

## Why Accelerate

RLCD needs a custom forward path: grouped perturbations are applied to option
logits, rewards come from a proper scoring rule, and the score-function loss is
evaluated under the live logit mean. Accelerate handles standard mechanics
without assuming token generation or a supervised scalar loss.

The generic contract carries `choice`, `score`, and `noul` semantics. The core
applies ranked probability score to ordered questions, while model adapters
remain responsible for native encoding and temperature persistence.

Qwen cannot reuse Laya's marker-before-state layout because causal tokens
cannot attend to later state tokens. The Qwen adapter repeats a complete
state/question/option-set sequence for every candidate and scores its final
token. This is correct for causal visibility but slower as option count grows.

## Why not Transformers Trainer

Trainer is suitable for ordinary supervised fine-tuning. It was used in the
previous SFT phase, but the main algorithm no longer has a supervised
cross-entropy objective. Wrapping RLCD as an opaque `compute_loss` would obscure
sampling statistics and strategy ownership without adding useful abstraction.

## Why not TRL GRPOTrainer

TRL GRPOTrainer assumes a causal language model that produces completion
tokens. Laya is a bidirectional decision encoder that scores a request-defined
option set in one forward pass. RLCD also groups probability reports for the
same labeled decision, while language-model GRPO groups generated sequences
and scores sequence rewards.

## Why not TorchRL in the default path

TorchRL's masked distributions are useful for episodic policy-gradient
ablations, which remain in the legacy package. The RLCD strategy explores
continuous logit perturbations and then evaluates complete probability
distributions; it does not sample one discrete environment action during
training. PyTorch softmax and masks are sufficient.

## Why not Unsloth

Unsloth targets supported generative LLM families, LoRA, quantization, and
CUDA-specific kernels. Laya is a custom ModernBERT decision architecture, and
Apple MPS is a required development target. A future causal-LM adapter may
choose Unsloth without changing the RLCD task contracts.

## Scaling boundary

Accelerate prepares model, optimizer, and dataloader for multiple processes.
Artifact writing and full distributed reproducibility still require explicit
multi-process acceptance tests. Use one process for current MPS and CUDA runs.
