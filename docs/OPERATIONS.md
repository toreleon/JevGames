# Operations and scaling

## Supported execution modes

### Apple Silicon MPS

The validated local environment is Apple M4 Pro with 48 GB unified memory.
Recommended settings:

- Python 3.12;
- float32 training;
- frozen model backbone;
- warm-up batch size 16;
- rollout inference batch size 32;
- online optimizer batch size 16;
- one MPS process.

Verify MPS before a long run:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.get_name())
PY
```

Accelerate currently provides device orchestration, but MPS does not become a
multi-GPU backend. Use one process.

### CUDA

CUDA is the intended scaling path. Install a PyTorch build matching the host
driver. Increase batches incrementally and validate numerical equivalence
before enabling mixed precision or unfreezing a backbone.

`accelerate launch` can manage multiple processes, but the current online
collector does not shard initial states per process. Multi-process execution
would duplicate rollout work. Implement rollout sharding before treating the
framework as distributed-efficient.

## Measured performance

On Apple M4 Pro 48 GB, a 100-level benchmark run measured:

| Phase | Work | Time |
|---|---:|---:|
| Setup | Load model and prepare 1,014 macros | 22.1 s |
| Expert warm-up | 1 epoch, batch 16 | 56.3 s |
| Online phase | 400 episodes, 1,686 decisions | 262.6 s |
| Save | ~804 MB checkpoint | 0.4 s |
| Total | Complete run | 341.5 s |

Linear planning estimates using the same frozen-backbone settings:

| Plan | Estimated time |
|---|---:|
| 2,000 levels, 6 warm-up epochs, 16,000 episodes | 4.8 hours |
| 10,000 levels, 6 warm-up epochs, 80,000 episodes | 24.0 hours |

Allow ±25% for trajectory length, OS contention, cache state, and thermal
throttling. These are capacity estimates, not service-level commitments.

## Performance tuning order

Change one dimension at a time:

1. confirm correctness with the smoke config;
2. raise rollout inference batch size;
3. raise warm-up and optimizer batch sizes;
4. lower gradient accumulation when memory allows;
5. increase rollout instance batch size;
6. profile tokenizer/collator overhead;
7. only then consider unfreezing encoder layers or mixed precision.

Do not increase group size merely for throughput. Group size changes advantage
statistics and exploration behavior.

## Monitoring

Current runs log phase summaries to stdout and write `report.json` at the end.
Monitor:

- warm-up loss and throughput;
- online solved episodes;
- group trainable-decision count;
- repeats and deadlocks;
- validation solve rate;
- elapsed phase time;
- system memory and thermal state.

A positive training reward trend without validation solves is not sufficient.

## Interruption and recovery

Optimizer resume is not implemented. If a process stops during warm-up or the
online phase, the final experiment checkpoint and report may not exist. Trainer
working files are not guaranteed to form a supported resume point because the
experiment orchestrator does not yet restore phase state.

For long jobs:

- run from a stable terminal/session;
- ensure sufficient disk before launch;
- retain the TOML and dataset manifest;
- redirect stdout through an external process supervisor if persistent logs are
  required;
- avoid system sleep;
- treat partial directories as incomplete unless `report.json` exists.

## Troubleshooting

### MPS unavailable

Confirm Apple Silicon, macOS support, and a native arm64 PyTorch wheel. Recreate
the project environment with `uv sync` rather than modifying system Python.

### Out of memory

Reduce, in order:

1. rollout inference batch size;
2. warm-up/online batch size;
3. rollout instance batch size;
4. context length in the model plugin;
5. number of simultaneously resident checkpoints.

Keep the backbone frozen. Do not enable mixed precision on MPS merely to hide
an unexplained memory leak.

### No trainable online decisions

This occurs when every episode in a group receives the same return or every
state is forced. Increase environment diversity, improve reward resolution, or
use a more competent warm-start policy.

### Training solves but validation does not

Likely causes include insufficient unique environments, leakage-resistant test
layouts being harder, overfitting a new action schema, or reward shaping that
does not track task completion. Increase unique training instances before
repeating episodes on the same few states.

### Immediate deadlocks

Verify task options, action masks, expert conversion, and canonical state
identity. In Sokoban, inspect push macros rather than primitive walking moves.

## Secrets and external services

Do not store Hub tokens or other credentials in TOML files or reports. Use the
environment or the platform's secret store. Checkpoints downloaded from public
registries are cached outside the repository by their client library.
