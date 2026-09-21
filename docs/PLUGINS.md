# Plugin API

Jev Games separates model behavior from task behavior through two abstract
contracts. A model plugin can be reused across tasks, and a task plugin can be
trained with different models.

The source of truth is [`jevgames/contracts.py`](../jevgames/contracts.py).

## Decision model adapter

A `DecisionModelAdapter` translates the framework's normalized decision
representation into a model-native batch.

### Required attributes

| Attribute | Meaning |
|---|---|
| `name` | Stable registry/model family name |
| `model` | Trainable `torch.nn.Module` or compatible wrapped object |
| `device` | Current PyTorch device |

### Required methods

#### `encode(observation, instruction, options, target_index=0)`

Encode one state and its dynamic choices. The return value is adapter-private,
but it must be consumable by `collate`.

Invariants:

- preserve option order;
- encode `target_index` without reordering options;
- reject an option set that exceeds model limits;
- do not silently truncate away complete options;
- use the same prompt/schema during training and inference.

#### `collate(items)`

Create a padded batch. The batch must contain everything needed by `logits`,
`action_mask`, `labels`, and `supervised_loss`.

#### `logits(active_model, batch)`

Return one logit per padded option. Calibration temperature or other fixed
inference transformations should be applied here so rollout and optimization
use the same distribution.

#### `action_mask(batch)`

Return a boolean tensor shaped like the logits. `True` means the option is
valid. Padded or invalid options must be `False`.

#### `labels(batch)`

Return the selected option index for each batch item.

#### `supervised_loss(active_model, batch)`

Return a scalar loss. The engine makes no assumptions about its components.
The built-in Laya adapter combines cross-entropy and proper-scoring loss.

#### `freeze_backbone()`

Freeze the model's representation backbone. Adapters without a backbone may
implement this as a no-op.

#### `save(model, destination, metadata)`

Save in a format loadable by the same adapter. Include framework metadata but
preserve the model family's native checkpoint structure.

### Minimal model skeleton

```python
from jevgames.contracts import DecisionModelAdapter
from jevgames.registry import register_model

class LinearAdapter(DecisionModelAdapter):
    name = "linear"

    def __init__(self, config):
        self.device = torch.device(config.get("device", "cpu"))
        self.model = MyNetwork().to(self.device)

    def encode(self, observation, instruction, options, target_index=0):
        ...

    def collate(self, items):
        ...

    def logits(self, active_model, batch):
        ...

    def action_mask(self, batch):
        ...

    def labels(self, batch):
        ...

    def supervised_loss(self, active_model, batch):
        ...

    def freeze_backbone(self):
        ...

    def save(self, model, destination, metadata):
        ...

@register_model("linear")
def create_linear(config):
    return LinearAdapter(config)
```

## Decision task

A `DecisionTask` owns dataset parsing, state representation, action semantics,
reward, terminal conditions, and state identity.

### Required attributes

| Attribute | Meaning |
|---|---|
| `name` | Stable task registry name |
| `instruction` | Instruction paired with every task decision |

### Required methods

#### `initial_states(dataset_path, limit=None)`

Load independent episode start states. Honor the optional limit without mixing
states from later trajectory steps.

#### `expert_decisions(dataset_path)`

Return `ExpertDecision` values. Each expert target must match one option key.
Episode IDs and steps should permit audit and trajectory reconstruction.

#### `observation(state)`

Return a serializable mapping suitable for model adapters. Avoid embedding
model-specific tokens or prompts here.

#### `options(state)`

Return unique `DecisionOption` values in deterministic order. An empty tuple
means no action is available. One option is treated as a forced transition.

#### `transition(state, option_key)`

Apply one legal decision and return `TaskTransition`, including reward,
termination, success, reason, and primitive-step count.

#### `serialize_state(state)` / `deserialize_state(value)`

Round-trip the opaque state for online policy updates. Serialization must be
stable throughout a run.

#### `state_key(state)`

Return a hashable canonical identity used for repeated-state detection. Merge
states only when they are behaviorally equivalent for future decisions.

#### `is_solved(state)`

Return task success independently of reward.

### Optional reward hooks

- `repeated_state_reward()` defaults to `-4.0`;
- `no_action_reward()` defaults to `-15.0`;
- `limit_reward()` defaults to `-1.0`.

### Minimal task skeleton

```python
from jevgames.contracts import DecisionTask
from jevgames.registry import register_task

class RoutingTask(DecisionTask):
    name = "routing"
    instruction = "Choose a destination queue"

    def initial_states(self, dataset_path, limit=None): ...
    def expert_decisions(self, dataset_path): ...
    def observation(self, state): ...
    def options(self, state): ...
    def transition(self, state, option_key): ...
    def serialize_state(self, state): ...
    def deserialize_state(self, value): ...
    def state_key(self, state): ...
    def is_solved(self, state): ...

@register_task("routing")
def create_routing(config):
    return RoutingTask(config)
```

## Registration and discovery

Built-in modules are imported lazily by `jevgames.registry`. Duplicate names
raise immediately.

Current third-party integration is explicit:

```python
import my_package.jev_games_plugins  # executes decorators
from jevgames.experiment import run_experiment
```

Automatic Python entry-point discovery is not implemented yet. Do not document
an external package as auto-discovered until this feature exists.

## Configuration

Once registered, model and task are selected independently:

```toml
[model]
type = "linear"
checkpoint = "models/linear-v1"

[task]
type = "routing"
reward_correct = 1.0

[data]
train = "data/routing/train.jsonl"
validation = "data/routing/validation.jsonl"
```

Plugin-specific keys are passed unchanged as a dictionary.

## Compatibility checklist

Before publishing a plugin:

- test `encode → collate → logits` shape consistency;
- test mask alignment after variable option counts;
- test state serialization round trips;
- test forced, empty, terminal, and repeated states;
- test checkpoint save and reload;
- run model-only held-out evaluation;
- document maximum context/options;
- document device and precision support;
- avoid importing another task or model plugin directly;
- pin any optional dependencies in a separate package extra.
