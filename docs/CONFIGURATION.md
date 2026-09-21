# Configuration reference

`jev-games run` accepts one TOML experiment. Relative paths are resolved from
the process working directory.

## `[experiment]`

| Field | Required | Meaning |
|---|---|---|
| `name` | no | Report name; defaults to the config filename |
| `output_dir` | no | Checkpoint and report destination |
| `seed` | no | Data order and RLCD exploration seed; default `42` |

## `[model]`

`type` is required and selects a registered model adapter. Other fields pass
to that adapter.

Built-in `laya` fields:

| Field | Default | Meaning |
|---|---|---|
| `source` | `convaiinnovations/laya` | Hub ID or local native checkpoint |
| `device` | `auto` | `auto`, `mps`, `cuda`, or `cpu` |
| `freeze_backbone` | `true` | Train decision layers while detaching encoder output |
| `gradient_checkpointing` | `true` | Enable when the backbone is trainable |

When `data.calibration` is present, the Laya adapter fits and persists scalar
temperatures per decision type and option-count bucket.

Built-in `qwen_decision` fields:

| Field | Default | Meaning |
|---|---|---|
| `source` | `Qwen/Qwen3-0.6B-Base` | Hub ID or native Jev Games Qwen checkpoint |
| `device` | `auto` | `auto`, `mps`, `cuda`, or `cpu` |
| `max_length` | `512` | Tokens retained per candidate-scoring sequence |
| `freeze_backbone` | `true` | Train only the decision head |
| `gradient_checkpointing` | `true` | Enable when the backbone is trainable |
| `dtype` | `auto` | Backbone storage dtype: CUDA BF16 when supported, otherwise FP32 |
| `lora_rank` | `0` | Positive rank enables PEFT LoRA and freezes the base weights |
| `lora_alpha` | `2 * lora_rank` | LoRA update scale |
| `lora_dropout` | `0.05` | Dropout inside LoRA branches |
| `lora_target_modules` | `all-linear` | PEFT target name or comma-separated module names |
| `backbone_learning_rate` | `0.00005` | Learning rate for LoRA/backbone parameters; the decision head uses `training.learning_rate` |

The Qwen adapter scores each candidate from the final token of a sequence that
contains the complete state, question, and option set. This preserves causal
visibility but repeats backbone work for every candidate.

When `lora_rank > 0`, Jev Games uses PEFT adapters even though
`freeze_backbone` defaults to `true`: the immutable base stays frozen, LoRA
weights remain trainable, and the decision head is trained separately. A native
checkpoint stores only the LoRA adapter plus the decision head and reloads the
base model named by `source`.

## `[task]`

`type` is required and selects a registered task. The built-in value is
`sokoban_push`. Its optional transition reward fields are retained for legacy
ablation tools; the main RLCD strategy does not read them.

## `[training]`

`type` is required. The built-in value is `rlcd`.

| Field | Default | Meaning |
|---|---:|---|
| `epochs` | `4` | Complete passes over calibration evidence |
| `update_epochs` | `1` | Replay passes after each environment collection |
| `batch_size` | `8` | Physical evidence rows per device batch |
| `gradient_accumulation` | `2` | Microbatches per optimizer synchronization |
| `group_size` | `4` | Perturbed distributions sampled per evidence row |
| `sigma_start` | `0.5` | Initial logit exploration standard deviation |
| `sigma_end` | `0.2` | Final logit exploration standard deviation |
| `learning_rate` | `0.0003` | AdamW learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `mixed_precision` | `no` | `no`, `fp16`, or `bf16` |
| `logging_steps` | `25` | RLCD microbatch log interval |

`group_size` must be at least two and both sigma values must be positive.
Apple MPS should start with `mixed_precision = "no"`. CUDA with BF16 support
should prefer `bf16`; otherwise use `fp16` only after a smoke comparison
confirms stable scores.

The removed `[warmup]` and `[online]` sections raise a configuration error.

## `[collection]`

This section is optional. `type = "episodic_outcomes"` enables online state
discovery and terminal-outcome evidence.

| Field | Default | Meaning |
|---|---:|---|
| `iterations` | `2` | collect/replay-update cycles |
| `episodes_per_instance` | `4` | stochastic episodes from each initial state |
| `max_instances` | all | cap on training initial states |
| `max_decisions` | `32` | episode decision horizon |
| `inference_batch_size` | `64` | per-action outcome questions per forward batch |
| `sampling_temperature` | `1.0` | temperature over predicted action success values |
| `epsilon` | `0.10` | uniform exploration mass over legal actions |

`epsilon` changes state exploration. RLCD `group_size` and `sigma` explore
probability reports for already observed questions; they do not discover states.

## `[data]`

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | Registered evidence provider; built-in `sokoban_solver` |
| `train` | yes | Training evidence source |
| `calibration` | no | Disjoint evidence used for post-training calibration fitting |
| `validation` | no | Held-out selection and diagnosis split |
| `test` | no | Held-out final reporting split |
| `manifest` | no | Dataset provenance and split-hash manifest |

The evidence plugin interprets these files. For `sokoban_solver` they are
solver trajectory JSONL files compressed into one-hot push decisions. The
recommended `sokoban_counterfactual_solver` plugin instead emits per-action
`noul` evidence aligned with environment control: expert successors are
positive, provable static-deadlock successors are negative, and unresolved
alternatives are not mislabeled. Laya needs at least ten examples in a
decision-type or option-count group before fitting a temperature for that
group.

## `[benchmark]`

| Field | Default | Meaning |
|---|---:|---|
| `max_decisions` | `32` | Greedy environment decision limit |
| `batch_size` | `32` | Inference batch size for both benchmark views |
| `max_instances` | all | Optional cap applied to environment levels |
| `max_evidence` | all | Optional separate cap for calibration decision rows |
| `calibration_bins` | `15` | Reliability bins used for expected calibration error |
| `before_training` | `false` | Record validation/test baselines before any update |

Level and decision caps are separate because one Sokoban level produces many
push decisions. Leaving `max_evidence` unset evaluates every held-out evidence
row even when `max_instances` bounds environment runtime.

## Validation behavior

Configuration parsing validates required plugin types and rejects legacy
training sections. Filesystem existence and plugin-specific field validation
happen when the experiment constructs the selected components.
