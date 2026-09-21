# Operations and scaling

## Apple Silicon MPS

Use Python 3.12, `device = "auto"` or `"mps"`, float32, a frozen backbone, and
one process. Start with batch 4–8 and group size 4. Raise the physical batch
only after watching unified-memory pressure.

```bash
uv run python - <<'PY'
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.get_name())
PY
```

## NVIDIA CUDA

Install a PyTorch build that recognizes the GPU architecture. Verify an actual
CUDA tensor operation before training. Start with the portable float32 config;
then test `mixed_precision = "fp16"` against identical smoke evidence before a
full run.

Full-backbone training consumes much more optimizer memory than the default
frozen-backbone run. Enable it only after measuring a representative batch.

## Capacity model

For `N` trainable decisions, `E` epochs, and group size `G`, RLCD scores:

```text
N × E × G probability distributions
```

The encoder forward pass is performed once per physical evidence batch; the
group dimension perturbs the smaller option-logit tensor. Runtime is therefore
not expected to grow linearly with `G` in the same way as full environment
episodes, but proper-score and sampling memory do grow with it.

Measure a 100-level run on the target hardware before projecting the 2,000-level
pilot. Old SFT/GRPO timings are not valid estimates for the new architecture.

## Tuning order

1. validate scores and gradients with the smoke config;
2. raise physical batch size until memory utilization is efficient;
3. tune gradient accumulation for the desired effective batch;
4. compare group sizes without changing sigma;
5. tune the sigma schedule using validation proper score and calibration;
6. only then unfreeze the backbone or enable mixed precision.

Group size and sigma change the estimator, not just throughput.

## Monitoring

Monitor RLCD proper score, loss, sigma, epoch time, held-out NLL/Brier/ECE,
environment solve rate, GPU utilization, memory, temperature, and disk. A
training proper-score increase is insufficient without held-out calibration.

## Interruption

Optimizer resume is not implemented. A terminated run may leave logs and an
incomplete output directory but no supported final checkpoint. Use a stable
terminal supervisor, retain the config and manifest, and keep the machine awake.

## Troubleshooting

- no trainable evidence: every state has only one legal option, or target data
  is invalid;
- zero advantages: sigma is too small or all sampled distributions receive the
  same numerical score;
- unstable gradients: reduce learning rate or sigma and keep float32;
- good accuracy with poor ECE/NLL: probabilities are overconfident;
- good calibration with low solve rate: the model is calibrated to a weak or
  mismatched target, or needs a planning component;
- CUDA Triton compiler error: install the host C build toolchain.
