# Paired objective experiment results — 2026-09-21

At the registered budget, direct proper-score optimization wins the paired test proper-score comparison. The tables below compare both trained methods with untrained, random and constant-prior controls; a forecasting win alone does not establish useful planning.

Fixed protocol: [OBJECTIVE_ABLATION.md](OBJECTIVE_ABLATION.md). These results concern this implementation, architecture, small dataset and update budget.

Three paired seeds [17, 29, 43]; 512 training rows from 64 fresh layouts, 128 optimizer updates per arm. Identical initialization hashes and update counts were verified. No checkpoint selection, online replay or temperature scaling was applied.

| Split | Levels | Evidence | Positive labels |
|---|---:|---:|---:|
| train | 64 | 512 | 150 |
| validation | 16 | 429 | 130 |
| test | 32 | 914 | 270 |

## Validation

Mean ± sample SD across three training seeds; higher proper score and solve rate are better, lower NLL/Brier are better. Rows from one layout are correlated.

| Predictor | Proper score | NLL | Brier | Solved levels per seed |
|---|---:|---:|---:|---|
| rlcd | -3.3360 ± 0.0955 | 3.6844 ± 0.0955 | 0.6061 ± 0.0000 | [2, 2, 2] |
| direct_proper_score | -0.2440 ± 0.0086 | 0.6217 ± 0.0067 | 0.4299 ± 0.0061 | [5, 4, 3] |
| Untrained same heads | -0.3505 ± 0.0923 | 0.7013 ± 0.0686 | 0.5079 ± 0.0680 | [4, 4, 5] |
| Constant train prior | -0.2337 | 0.6137 | 0.4226 | — |
| Uniform random actions | — | — | — | [2, 3, 0] |

Paired proper-score differences (RLCD − direct): [-3.06243, -3.19252, -3.020918].


## Test

Mean ± sample SD across three training seeds; higher proper score and solve rate are better, lower NLL/Brier are better. Rows from one layout are correlated.

| Predictor | Proper score | NLL | Brier | Solved levels per seed |
|---|---:|---:|---:|---|
| rlcd | -3.2387 ± 0.0918 | 3.5910 ± 0.0918 | 0.5908 ± 0.0000 | [5, 7, 5] |
| direct_proper_score | -0.2380 ± 0.0104 | 0.6172 ± 0.0081 | 0.4256 ± 0.0074 | [5, 7, 5] |
| Untrained same heads | -0.3504 ± 0.0957 | 0.7013 ± 0.0711 | 0.5079 ± 0.0705 | [7, 5, 7] |
| Constant train prior | -0.2249 | 0.6069 | 0.4163 | — |
| Uniform random actions | — | — | — | [4, 10, 6] |

Paired proper-score differences (RLCD − direct): [-2.971892, -3.095777, -2.9342810000000004].

## Registered criterion

RLCD promise criterion met: **False**. Mean proper-score difference -3.0007, mean Brier difference 0.1652.

Mean solved-level difference (RLCD − direct): 0.00/32. The separate gameplay threshold (+2 levels) was met: **False**.

## Execution receipts

| Method | Seed | Updates | Training seconds | Peak allocated GiB | Reload max probability error |
|---|---:|---:|---:|---:|---:|
| rlcd | 17 | 128 | 85.5 | 1.677 | 1.27e-07 |
| rlcd | 29 | 128 | 85.9 | 1.677 | 2.29e-07 |
| rlcd | 43 | 128 | 87.6 | 1.678 | 4.61e-07 |
| direct_proper_score | 17 | 128 | 82.6 | 1.677 | 0.00207 |
| direct_proper_score | 29 | 128 | 83.6 | 1.677 | 0.00206 |
| direct_proper_score | 43 | 128 | 84.2 | 1.678 | 0.00132 |

Each saved checkpoint was reloaded and used for one held-out episode. Per-seed reports preserve per-layout forecasts and gameplay outcomes. A constant-prior win alone is not evidence of spatial reasoning. The experiment uses small one/two-box levels; conclusions do not extend to hard Sokoban or independently tuned versions of either objective.

## Precision and attribution limits

Training and post-training evaluation use the Accelerate BF16 forward wrapper. Native reload uses BF16 backbone storage with FP32 heads, and the checkpoint serializes heads as FP16. A small probability error can change action ordering when candidate scores nearly tie. Reload is therefore a functional/approximate numerical check, not proof of bit-identical trajectories.

Arms whose checked episode trace differed after reload: [('rlcd', 17), ('rlcd', 29), ('rlcd', 43), ('direct_proper_score', 29)].

The large raw calibration gap should not be confused with evidence that either model has learned useful planning. Treat small solved-level differences as exploratory, given the small sample and precision sensitivity.

A separate [constant-input diagnostic](ADVANTAGE_NORMALIZATION_DIAGNOSTIC.md) isolates a calibration failure of the per-example reward standardization. It was added after the registered study began, without changing these runs.
