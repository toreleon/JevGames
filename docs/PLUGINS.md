# Plugin API

Jev Games has independent model, task, evidence, and training-strategy
registries. A TOML experiment selects one of each.

## Model adapters

Implement `DecisionModelAdapter` and register a factory:

```python
from jevgames.contracts import DecisionModelAdapter
from jevgames.registry import register_model

class MyModel(DecisionModelAdapter):
    name = "my_model"
    # encode, collate, logits, action_mask, target_probabilities,
    # calibration_reward, freeze_backbone, save

@register_model("my_model")
def create_my_model(config):
    return MyModel(config)
```

The key methods are:

- `encode`: convert an observation, instruction, dynamic options, and optional
  target distribution into one native item;
- `collate`: pad native items into a batch;
- `logits`: return one logit per padded option;
- `action_mask`: mark real options and exclude padding;
- `target_probabilities`: expose the batch targets;
- `calibration_reward`: score reported distributions with a strictly proper
  rule and return one reward per distribution and evidence item;
- `save`: preserve the model's native checkpoint format.

`calibration_reward` must support probabilities shaped `[group, batch, option]`
during RLCD and `[batch, option]` during evaluation.

## Task adapters

Implement `DecisionTask` and register a factory. The task exposes environment
state, legal options, transitions, and terminal checks for model-only evaluation.

## Evidence providers

Implement `EvidenceProvider.load(dataset_path, task)` and register it with
`register_evidence`. It converts a dataset into `CalibrationDecision` rows for
a compatible task. This keeps file format, labeling policy, and provenance
independent from environment mechanics.

`CalibrationDecision.target_probabilities` may be one-hot or soft. Record a
meaningful `provenance` value such as `solver_solution`,
`environment_outcomes`, or `teacher_distribution` so reports can audit the
source.

Task adapters should define canonical state identity carefully. Sokoban, for
example, treats all player positions within one reachable region as equivalent
when the boxes do not move.

## Training strategies

Implement `TrainingStrategy` and register it:

```python
from jevgames.contracts import TrainingStrategy
from jevgames.registry import register_strategy

class MyStrategy(TrainingStrategy):
    name = "my_strategy"

    def train(self, adapter, items, output_dir, seed):
        ...
        return [{"phase": "my_strategy", "metric": 1.0}]

@register_strategy("my_strategy")
def create_my_strategy(config):
    return MyStrategy(config)
```

A strategy operates through the model contract and must not import a particular
task. Task-specific data collection should happen before the evidence reaches
the strategy.

## Registration and discovery

Built-ins are loaded lazily by `jevgames.registry`. Run:

```bash
uv run jev-games plugins
```

Third-party Python entry-point discovery is not implemented yet. External
plugins must currently be imported by an integration package before registry
lookup.

## Contract invariants

- option keys are unique within a decision;
- target length equals option count and sums to one;
- padding is always false in the action mask;
- forced decisions are excluded from training;
- a proper score rewards honest probability distributions in expectation;
- validation and test evidence never enters `TrainingStrategy.train`;
- checkpoint metadata names the selected model, task, evidence, and strategy.
