# Plugin API

Jev Games has independent model, task, evidence, collector, and
training-strategy registries.

## Model adapters

Implement `DecisionModelAdapter` and register a factory:

```python
from jevgames.contracts import DecisionModelAdapter
from jevgames.registry import register_model

class MyModel(DecisionModelAdapter):
    name = "my_model"
    # encode, collate, logits, training_logits, action_mask, target_probabilities,
    # ordinal_mask, freeze_backbone, fit_calibration, save

@register_model("my_model")
def create_my_model(config):
    return MyModel(config)
```

The key methods are:

- `encode`: convert an observation, `DecisionQuestion`, and optional target
  distribution into one native item;
- `collate`: pad native items into a batch;
- `logits`: return one logit per padded option;
- `training_logits`: optionally bypass fitted inference calibration transforms;
- `action_mask`: mark real options and exclude padding;
- `target_probabilities`: expose the batch targets;
- `ordinal_mask`: identify `score` rows for ranked probability scoring;
- `fit_calibration`: optionally fit model-native post-training calibration
  parameters from a dedicated split;
- `save`: preserve the model's native checkpoint format.

The composite proper-scoring rule belongs to `jevgames.scoring`, so a new model
cannot accidentally change what the `rlcd` strategy optimizes.

## Task adapters

Implement `DecisionTask` and register a factory. The task exposes environment
state, a typed `DecisionQuestion`, transitions, and terminal checks for
model-only evaluation.

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
task.

## Evidence collectors

Implement `EvidenceCollector.collect` to interact with a task and return
`CollectedEvidence`. A collector owns state exploration and labeling policy;
the training strategy continues to own probability-distribution optimization.

The built-in `episodic_outcomes` collector requires per-action `noul` questions
from `DecisionTask.outcome_queries`. It labels selected actions with terminal
episode outcomes and records its sampling temperature and epsilon.

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
- questions declare `choice`, ordered `score`, or false/true `noul` semantics;
- target length equals option count and sums to one;
- padding is always false in the action mask;
- forced decisions are excluded from training;
- a proper score rewards honest probability distributions in expectation;
- validation and test evidence never enters `TrainingStrategy.train`;
- checkpoint metadata names the selected model, task, evidence, collector, and strategy.
