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

## `[task]`

`type` is required and selects a registered task. The built-in value is
`sokoban_push`. Its optional transition reward fields are retained for legacy
ablation tools; the main RLCD strategy does not read them.

## `[training]`

`type` is required. The built-in value is `rlcd`.

| Field | Default | Meaning |
|---|---:|---|
| `epochs` | `4` | Complete passes over calibration evidence |
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
Apple MPS should start with `mixed_precision = "no"`. CUDA may use `fp16`
after a smoke comparison confirms stable scores.

The removed `[warmup]` and `[online]` sections raise a configuration error.

## `[data]`

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | Registered evidence provider; built-in `sokoban_solver` |
| `train` | yes | Training evidence source |
| `validation` | no | Held-out selection and diagnosis split |
| `test` | no | Held-out final reporting split |
| `manifest` | no | Dataset provenance and split-hash manifest |

The evidence plugin interprets these files. For `sokoban_solver` they are
solver trajectory JSONL files compressed into push decisions.

## `[benchmark]`

| Field | Default | Meaning |
|---|---:|---|
| `max_decisions` | `32` | Greedy environment decision limit |
| `batch_size` | `32` | Inference batch size for both benchmark views |
| `max_instances` | all | Optional cap applied to evidence and environments |
| `calibration_bins` | `15` | Reliability bins used for expected calibration error |

## Validation behavior

Configuration parsing validates required plugin types and rejects legacy
training sections. Filesystem existence and plugin-specific field validation
happen when the experiment constructs the selected components.
