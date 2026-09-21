# Benchmarking protocol

Jev Games benchmarks answer one question: can the saved model make successful
decisions on states and environments that were not used for training?

## Model-only rule

Benchmark evaluation is model-only. Do not enable:

- expert labels;
- A* or another search solver;
- low-confidence fallback;
- retry with a different policy;
- manual intervention;
- data from validation or test during training.

Task-defined deterministic action execution is allowed. For example, the
Sokoban task lets the model choose a box push and uses BFS to execute the
walking path. BFS does not choose the strategic push and is therefore part of
the environment interface rather than a fallback solver.

## Dataset splits

Split at the environment-instance level, not at the individual transition
level. States from one trajectory must never appear across train and validation
or test.

For generated environments, use a content identity that captures the factors
that would allow memorization. The Sokoban generator uses wall-and-goal layout
hashes and enforces disjoint layouts across all splits.

Recommended minimum pilot split:

| Split | Instances | Purpose |
|---|---:|---|
| Train | 2,000 | Warm-up and online collection |
| Validation | 200 | Configuration and checkpoint selection |
| Test | 200 | Final held-out estimate |

Do not tune reward coefficients or stopping thresholds against the test split.

## Generic metrics

The generic evaluator currently reports:

| Metric | Definition |
|---|---|
| `instances` | Number of evaluated initial states |
| `solved` | Instances ending in task success |
| `solve_rate` | `solved / instances` |
| `mean_environment_steps` | Mean strategic decisions executed |
| `mean_primitive_steps` | Mean underlying primitive transitions |
| `outcomes` | Counts by terminal reason |

The task can expose richer metrics through a task-specific benchmark module.
The Sokoban compatibility benchmark additionally supports per-difficulty
breakdowns and mean pushes.

## Running benchmarks

An experiment automatically evaluates configured validation and test datasets:

```bash
uv run jev-games run configs/sokoban_smoke.toml
```

Results are written under `benchmarks` in `report.json`.

The Sokoban-specific compatibility command can emit a standalone report:

```bash
uv run python -m sokoban_laya.cli benchmark-macro \
  runs/example/checkpoint \
  data/pilot/validation.jsonl \
  --manifest data/pilot/manifest.json \
  --output runs/example/benchmark-validation.json
```

## Comparison protocol

When comparing two models or algorithms:

1. use byte-identical validation and test datasets;
2. use the same maximum decision budget;
3. use the same task reward and transition implementation;
4. record seeds and dependency lockfile;
5. compare model-only solve rate first;
6. report deadlocks, repeats, no-action failures, and limits;
7. report at least three training seeds for a robust claim;
8. keep validation-based selection separate from final test reporting.

Training reward is not a substitute for held-out solve rate. A policy can
improve shaped reward while remaining unable to complete a level.

## Confidence and calibration

Confidence is useful only after domain calibration has been evaluated. Do not
use a confidence threshold in the primary model-only benchmark. A separate
deployment benchmark may measure coverage versus accuracy under escalation or
fallback, but it must be labeled as a hybrid-system result.

## Existing measured baselines

The 100-level framework benchmark used one expert epoch and one online
iteration. It produced non-zero training episode solves but 0/20 held-out
validation solves. This validates the pipeline and demonstrates that a small
run does not establish generalization.

The Microban 3 case study can be solved after task-specific macro warm-up and
online optimization, but that level participated in training. It is evidence
of learning and execution correctness, not held-out generalization.

## Report interpretation

Example:

```json
{
  "instances": 200,
  "solved": 76,
  "solve_rate": 0.38,
  "mean_environment_steps": 9.42,
  "mean_primitive_steps": 31.77,
  "outcomes": {
    "solved": 76,
    "static_deadlock": 83,
    "repeat": 31,
    "decision_limit": 10
  }
}
```

This should be read as a 38% model-only solve rate under the configured budget,
not as a claim that the policy solves arbitrary instances from the task family.
