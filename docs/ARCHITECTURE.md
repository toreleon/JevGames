# Architecture

## Design goals

Jev Games is designed around four constraints:

1. **Model independence.** The engine must not know how a model tokenizes an
   observation, represents choices, calculates a supervised loss, or stores a
   checkpoint.
2. **Task independence.** The engine must not know Sokoban rules, reward
   shaping, dataset schemas, or what constitutes a terminal state.
3. **Dynamic decisions.** Available actions may vary at every state and may be
   described by text rather than fixed numeric IDs.
4. **Framework ownership.** Standard training mechanics should be delegated to
   maintained libraries; project code should implement only domain-specific
   behavior and the group-relative objective.

## Component model

```text
┌──────────────────────────────────────────────────────────────┐
│ ExperimentConfig                                             │
│ model.type  task.type  data  warmup  online  benchmark       │
└────────────────────────────┬─────────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
┌────────────▼─────────────┐      ┌──────────▼──────────────┐
│ DecisionModelAdapter    │      │ DecisionTask            │
│                        │      │                         │
│ encode / collate       │      │ expert_decisions        │
│ logits / action_mask   │      │ initial_states          │
│ supervised_loss        │      │ observation / options   │
│ freeze_backbone / save │      │ transition / reward     │
└────────────┬────────────┘      │ state_key / terminal    │
             │                   └──────────┬──────────────┘
             └───────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Generic engine │
                    └────────┬────────┘
         ┌───────────────────┼────────────────────┐
         │                   │                    │
┌────────▼─────────┐ ┌───────▼──────────┐ ┌──────▼──────────┐
│ Supervised phase │ │ Online GRPO      │ │ Evaluation      │
│ HF Trainer       │ │ Accelerate       │ │ Greedy, no      │
│                  │ │ TorchRL masking  │ │ fallback        │
└──────────────────┘ └──────────────────┘ └─────────────────┘
```

## Dependency boundaries

The `jevgames` package is the framework boundary.

- `jevgames.engine` imports only contracts and training frameworks.
- `jevgames.models.*` may import model-specific libraries and checkpoint code.
- `jevgames.tasks.*` may import environment and dataset implementations.
- model plugins must not import task plugins;
- task plugins must not import model plugins;
- the registry is the only construction mechanism used by experiments.

The current Laya and Sokoban plugins reuse implementation from
`sokoban_laya`, which is retained as a compatibility package. New framework
code should depend on the adapters, not directly on that package.

## Normalized decision representation

Every task exposes a tuple of `DecisionOption` values:

```python
DecisionOption(
    key="box_r2_c6_push_left",
    description="Push the box at row 3, column 7 one square left",
)
```

The option key must be unique within the state. The description is intended
for models that jointly encode option semantics and state context. A numeric
policy can ignore the description inside its adapter.

Forced states with exactly one option are executed by the environment without
calling the model. This avoids meaningless gradients and accommodates models
whose uncertainty features require at least two choices.

## Experiment lifecycle

`run_experiment` is the orchestration boundary:

```text
load config
  → construct adapters
  → load expert decisions
  → encode trainable examples
  → supervised warm-up
  → collect grouped online episodes
  → clipped group-relative updates + expert replay
  → save native checkpoint
  → validation benchmark
  → test benchmark
  → report.json
```

Validation and test evaluation are model-only. Search, oracle labels, or safety
fallbacks must not be inserted into the benchmark path.

## Framework responsibilities

### Transformers Trainer

Trainer owns supervised dataloading, shuffling, optimizer construction,
gradient accumulation, clipping, logging, and epoch scheduling. The model
adapter supplies the actual loss.

### Accelerate

Accelerate owns device placement and synchronized online optimization. The
current MPS path is single-process. CUDA multi-process execution requires
rollout sharding before it should be considered efficient.

### TorchRL

`MaskedCategorical` owns legal-action normalization, stochastic sampling,
log-probability, entropy, and greedy mode selection.

## Registry lifecycle

Built-ins are loaded lazily the first time the registry is queried. A plugin
factory receives its untyped TOML table and returns an object satisfying the
corresponding abstract contract.

Current registry discovery is explicit. Third-party packages must import their
registration module before creating an experiment. Python entry-point discovery
is listed as future work.

## Package layout

```text
jevgames/
├── cli.py
├── config.py
├── contracts.py
├── engine.py
├── experiment.py
├── registry.py
├── models/
│   └── laya.py
└── tasks/
    └── sokoban.py
```

Task-specific generators, solvers, and visual controllers do not belong in the
generic package. They live in the task implementation package.
