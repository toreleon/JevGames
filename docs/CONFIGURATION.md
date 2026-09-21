# Configuration reference

Experiments are defined in TOML and loaded by `jevgames.config`. Paths are
currently resolved relative to the process working directory, normally the
repository root. They are not resolved relative to the TOML file.

Unknown fields in typed framework sections fail during dataclass construction.
Model and task plugin tables accept plugin-defined fields.

## `[experiment]`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `name` | string | configuration filename | Human-readable run name |
| `output_dir` | string | `runs/<config-name>` | Reports and checkpoint root |
| `seed` | integer | `42` | Trainer, rollout, and sampling seed |

## `[model]`

The `type` field selects a registered `DecisionModelAdapter`. Remaining fields
are passed unchanged to the adapter factory.

| Field | Required | Meaning |
|---|---:|---|
| `type` | yes | Registry name, currently `laya` |

### Built-in `laya` fields

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `source` | string | `convaiinnovations/laya` | Local checkpoint or Hub model ID |
| `device` | string | `auto` | `auto`, `mps`, `cuda`, or `cpu` |
| `freeze_backbone` | boolean | `true` | Freeze the transformer encoder |

Freezing the backbone is the supported M4 starting point. Unfreezing greatly
increases memory, optimizer state, and runtime requirements.

## `[task]`

The `type` field selects a registered `DecisionTask`. Remaining fields are
passed to the task factory.

| Field | Required | Meaning |
|---|---:|---|
| `type` | yes | Registry name, currently `sokoban_push` |

### Built-in `sokoban_push` fields

| Field | Type | Default | Meaning |
|---|---:|---:|---|
| `solve_reward` | float | `50.0` | Terminal success reward |
| `goal_reward` | float | `4.0` | Reward for a box newly placed on a goal |
| `off_goal_penalty` | float | `4.0` | Penalty for removing a box from a goal |
| `push_penalty` | float | `0.02` | Cost per macro push |
| `walk_penalty` | float | `0.002` | Cost per primitive walking step inside a macro |
| `deadlock_penalty` | float | `15.0` | Static-deadlock/no-action penalty |
| `repeat_penalty` | float | `4.0` | Repeated canonical-state penalty |
| `limit_penalty` | float | `1.0` | Decision-limit penalty |

## `[data]`

| Field | Required | Meaning |
|---|---:|---|
| `train` | yes | Expert dataset and online initial-state source |
| `validation` | no | Held-out states evaluated after training |
| `test` | no | Final held-out states evaluated after validation |
| `manifest` | no | Dataset metadata; retained in experiment config/report |

The generic engine does not parse dataset contents. The selected task adapter
owns the schema and loading logic.

## `[warmup]`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `epochs` | integer | `6` | Expert warm-up epochs; `0` disables the phase |
| `batch_size` | integer | `16` | Per-device Trainer batch size |
| `gradient_accumulation` | integer | `2` | Trainer accumulation steps |
| `learning_rate` | float | `0.0003` | Warm-up optimizer learning rate |
| `weight_decay` | float | `0.01` | AdamW weight decay |
| `max_grad_norm` | float | `1.0` | Gradient clipping norm |
| `logging_steps` | integer | `25` | Trainer logging interval |

The model adapter defines the supervised loss. For Laya it combines
cross-entropy with a proper-scoring calibration term.

## `[online]`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `iterations` | integer | `2` | Collect/update passes; `0` disables online RL |
| `group_size` | integer | `4` | Episodes sampled from each initial state |
| `max_decisions` | integer | `32` | Environment decisions per episode |
| `max_instances` | integer/null | all | Limit training initial states |
| `batch_size` | integer | `16` | Online optimizer batch size |
| `learning_rate` | float | `0.0003` | Online AdamW learning rate |
| `gradient_accumulation` | integer | `2` | Accelerate accumulation steps |
| `weight_decay` | float | `0.01` | Online AdamW weight decay |
| `max_grad_norm` | float | `1.0` | Accelerate clipping norm |
| `rollout_instance_batch_size` | integer | `16` | Initial states processed together |
| `rollout_inference_batch_size` | integer | `32` | Model inference batch during collection |
| `clip_ratio` | float | `0.2` | PPO-style probability-ratio clipping |
| `entropy_weight` | float | `0.01` | Exploration entropy coefficient |
| `expert_weight` | float | `0.4` | Expert replay loss coefficient |

Total episodes per iteration are approximately:

```text
min(training instances, max_instances) × group_size
```

Groups with identical returns produce zero group-relative advantage and do not
contribute online policy records.

## `[benchmark]`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `max_decisions` | integer | `32` | Greedy decisions allowed per instance |
| `batch_size` | integer | `32` | Benchmark inference batch size |
| `max_instances` | integer/null | all | Optional bounded evaluation subset |

Benchmark evaluation is greedy and model-only. Task transitions still execute
deterministic action macros where the task defines them.

## Validation behavior

Configuration loading validates required `model.type`, `task.type`, and
`data.train` fields. It does not currently perform filesystem existence checks,
range validation, or cross-field validation before plugin construction. Those
checks are roadmap items; long jobs should begin with the smoke configuration.
