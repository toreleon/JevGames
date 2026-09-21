# Artifacts and schemas

## Experiment directory

```text
runs/<experiment>/
├── checkpoint/             # model-native files
└── report.json             # framework report, schema version 4
```

The Laya checkpoint contains `model.safetensors`, `rl_agent_config.json`, an
encoder config, tokenizer files, and `training_stats.json`.

## Report schema

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | currently `4` |
| `name` | experiment identity |
| `config` | fully parsed experiment config |
| `plugins` | selected model, task, evidence, collector, and strategy plugins |
| `stats` | setup, RLCD, collection, and post-training calibration records |
| `benchmarks` | validation/test calibration and environment results |
| `checkpoint` | exported model path |
| `elapsed_seconds` | complete experiment duration |

Each split has this shape:

```json
{
  "validation": {
    "calibration": {
      "instances": 100,
      "accuracy": 0.71,
      "brier_score": 0.34,
      "expected_calibration_error": 0.08,
      "reliability_bins": []
    },
    "environment": {
      "instances": 20,
      "solved": 4,
      "solve_rate": 0.2,
      "outcomes": {"solved": 4, "static_deadlock": 16}
    },
    "outcome_calibration": {
      "instances": 80,
      "brier_score": 0.21,
      "expected_calibration_error": 0.06
    }
  }
}
```

Numbers above illustrate shape only.

## Evidence schema

The public Sokoban JSONL stores primitive solver decisions. The evidence plugin
compresses those records into in-memory `CalibrationDecision` rows. A future
general evidence format should preserve target distributions, provenance,
sampling budget, unresolved outcomes, option identity, and environment hash.

## Manifests

Dataset manifests should record generator version, seed, split counts, layout
hashes, difficulty, and evidence provenance. Calibration layouts must also be
disjoint from train, validation, and test. Never infer held-out status from a
filename alone.

## Compatibility

Schema version 1 reports contain `supervised_warmup` and `online_grpo` stats.
Schema version 2 introduced RLCD but had choice-only contracts and no dedicated
post-training calibration stage. Schema version 3 added typed questions and
temperature fitting but no online collector. Consumers must branch on
`schema_version`.
