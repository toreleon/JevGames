# Registered Sokoban objective comparison — 2026-09-21

## Question and limits

Does the current Jev Games implementation of Gaussian-report RLCD provide
better probability forecasts or control than directly differentiating the
same composite proper score, at a fixed update budget?

This tests our Laya-inspired RLCD implementation, not a reproduction of
TypeSafe's unpublished training stack. Group advantage normalization and
Gaussian smoothing change the training objective: this is a comparison of
two training recipes, not two unbiased estimators of precisely the same loss.
Hyperparameters are fixed, not individually optimized for either method.

## Protocol fixed before GPU runs

- Fresh 6x6 procedural layouts, alternating one and two boxes. Layouts cannot
  overlap across train (64), validation (16) and test (32); known previous
  pilot manifests are excluded as well.
- Exhaustively construct each reachable push-macro graph up to 2,500 states.
  Prune only provable static-deadlock states. Reverse BFS supplies exact
  shortest **push** distances; walking does not consume the budget. Reject an
  entire layout when exploration exceeds the state budget.
- Event: after the specified action, does any continuation solve within H
  total pushes **including the first push**? H is part of the model input.
  Train on root and two other states with budgets near their exact distance.
  All executable actions receive verified labels, including safe detours and
  immediate deadlocks. No oracle values appear in the observation.
- Deterministically subsample 512 training rows, without rebalancing labels.
  Retain complete action groups in validation/test; report class balance.
- Qwen3-0.6B-Base, rank-8 LoRA, binary NOUL head, BF16, activation checkpointing.
  Three paired seeds (17, 29, 43). Each pair has identical initialization
  (checked by hash), minibatch order, dropout RNG stream and optimizer settings.
- Four epochs of 32 batches: exactly 128 optimizer updates per arm. Head LR
  3e-5 and backbone LR 1e-5. Fixed final checkpoint; no validation selection,
  post-hoc temperature fit, online replay, early stopping or tuning on test.
- RLCD: G=16, sigma 0.6 to 0.15. Direct: negative mean of the identical
  log+spherical proper score. Shared Accelerate training loop.
- Compare raw NLL, Brier, proper score, balanced accuracy, ECE and action
  ranking. Binary classification labels are deterministic, so confidence alone
  is not evidence of epistemic uncertainty calibration.
- Gameplay uses the same greedy controller, eight push budget, all legal
  actions, no search/deadlock masking/novelty bonus. The budget counts down;
  revisits are counted, not forcibly stopped. Random and untrained baselines
  use the same levels and controller rules.
- A constant training-prevalence predictor is another forecasting baseline;
  beating a random head alone can reflect learning the class prior, not learning
  the board. Only improvements over this baseline count as informative forecasts.

## Decision rule

Primary endpoint is mean held-out composite proper score (higher is better).
Treat RLCD as *promising at this budget*, not universally better, only if its
paired advantage is positive at all three seeds and at least 0.02 on average,
with no mean Brier degradation. A gameplay benefit additionally requires at
least two more solved test levels on average (2/32 absolute solve-rate gain).
Report any disagreements among calibration, ranking and gameplay metrics.
If neither method beats its untrained baseline meaningfully, conclude that
this budget/representation is inconclusive; do not infer RLCD impossibility.
Three seeds and correlated decisions within layouts do not support broad
statistical claims. Retain per-layout and per-seed results; test remains a
diagnostic for this frozen protocol, not a future tuning set.

The new dedicated binary head changed both representation and training
behavior in the previous pilot, so that pilot cannot establish RLCD's advantage
over a direct objective. Its high binary accuracy also did not improve gameplay.

## Reproduction

```bash
uv sync --locked --extra qwen --extra train --python 3.12
uv run python -m jevgames.research.objective_ablation prepare configs/sokoban_objective_ablation.toml
uv run python -m jevgames.research.objective_ablation run configs/sokoban_objective_ablation.toml
```

The runner refuses to overwrite an existing arm or use a changed registered
config. Each arm runs in its own process and saves baseline, final report,
checkpoint, timing, optimizer update count, initialization hash and a native
reload check. Dataset SHA-256 hashes and graph exclusions are saved in the
manifest. Large generated files stay under ignored `runs/`; protocol and
implementation are versioned. Failures require inspection, not automatic retry.

Outputs: `runs/sokoban-objective-ablation-20260921/summary.json`, six per-arm
reports/checkpoints/logs, `random_baselines.json`, and `data/manifest.json`.
