# Artifacts and schemas

## Experiment directory

Every config-driven run writes to `experiment.output_dir`:

```text
runs/<experiment>/
├── checkpoint/
│   ├── model-native files
│   └── training_stats.json
├── warmup/                  # Trainer working directory, when created
└── report.json
```

The model adapter owns checkpoint filenames. The built-in Laya adapter writes
Safetensors weights, encoder config, tokenizer files, and
`rl_agent_config.json`.

## `report.json`

Top-level schema version is currently `1`:

```json
{
  "schema_version": 1,
  "name": "laya-sokoban-smoke",
  "config": {},
  "plugins": {
    "model": "laya",
    "task": "sokoban_push"
  },
  "stats": [],
  "benchmarks": {
    "validation": {},
    "test": {}
  },
  "checkpoint": "runs/example/checkpoint",
  "elapsed_seconds": 29.523
}
```

The embedded config is the resolved dataclass representation, not the original
TOML text. Retain the source TOML separately when provenance matters.

## Training statistics

`stats` contains ordered phase records. Current phase values include:

- `setup`;
- `supervised_warmup`;
- `online_grpo`.

Online records include episode count, solve count, mean reward, number of
trainable decisions, outcome counts, and elapsed seconds.

The built-in Laya checkpoint also stores framework metadata in
`training_stats.json`. Model plugins may use a different native metadata file.

## Generic benchmark schema

```json
{
  "instances": 20,
  "solved": 0,
  "solve_rate": 0.0,
  "mean_environment_steps": 4.4,
  "mean_primitive_steps": 16.8,
  "outcomes": {
    "no_action": 4,
    "repeat": 2,
    "static_deadlock": 14
  }
}
```

Task-specific compatibility benchmarks may add dimensions such as difficulty,
push count, calibration, or fallback coverage.

## Sokoban expert JSONL

The current Sokoban task consumes primitive solver records:

```json
{
  "board": "#####\n#@$.#\n#####",
  "legal_actions": ["push_right"],
  "expert_action": "push_right",
  "episode": "train_000001_abcd",
  "step": 0
}
```

The task adapter groups rows by episode and compresses primitive trajectories
to push-macro expert decisions. This schema belongs to the Sokoban plugin, not
the generic framework.

## Sokoban dataset manifest

`scripts/generate_curriculum.py` writes:

```text
data/pilot/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

Each manifest record contains episode ID, layout and level hashes, difficulty,
seed, dimensions, box count, reverse-push count, and expert trajectory length.
The generator enforces disjoint layout hashes across splits.

## Model size and disk planning

The built-in Laya checkpoint is approximately 804 MB in fp16 Safetensors form.
Every retained run can therefore consume close to 1 GB after metadata. Dataset
and report files are comparatively small.

Before long sweeps:

- estimate checkpoint count;
- choose a retention policy;
- preserve reports before archiving checkpoints;
- never delete the only checkpoint supporting a reported result;
- avoid storing cache downloads inside version control.

## Versioning

Schema fields may expand within version `1`; consumers should ignore unknown
fields. A breaking semantic or structural change should increment
`schema_version` and include a migration note.

No dataset or report schema registry exists yet. Plugin authors should document
their source format and version inside their package.
