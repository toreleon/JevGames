# Benchmarking protocol

Jev Games evaluates two properties separately: whether reported probabilities
are calibrated and whether the composed policy solves complete environments.

## Split discipline

Train, calibration, validation, and test must be disjoint by underlying
environment layout, not merely by serialized state. Sokoban manifests record
layout and level hashes so trajectories from one layout cannot cross splits.

Weight training reads only `data.train`; temperature fitting reads only
`data.calibration`. Validation and test are used for reporting after both.

## Calibration benchmark

The model receives held-out observations and dynamic option sets. It reports a
masked probability distribution, which is compared with the held-out target.

The report includes:

| Metric | Meaning |
|---|---|
| `accuracy` | argmax option matches argmax target |
| `soft_accuracy` | target probability assigned to the predicted option |
| `negative_log_likelihood` | cross-entropy against the target distribution |
| `brier_score` | sum of squared multiclass probability errors |
| `expected_calibration_error` | weighted confidence/accuracy gap across bins |
| `mean_top_probability` | mean probability of the argmax option |
| `mean_entropy_confidence` | mean `1 - H(p)/log(K)` confidence |
| `mean_proper_score` | framework composite proper score |
| `score_mean_absolute_error` | expected-level error for ordinal questions |
| `selective_accuracy` | accuracy when acting on the most confident 100/80/50% |
| `reliability_bins` | count, confidence, accuracy, and gap per occupied bin |

ECE depends on binning and sample count. Always inspect reliability bins and
NLL/Brier alongside it.

For soft targets, reliability accuracy is the target mass assigned to the
predicted option. Treating `argmax(target)` as certain would incorrectly report
non-zero ECE when prediction equals the target distribution.

## Environment benchmark

The saved model plays greedily with no solver fallback and no weight updates.
For tasks with outcome queries, every legal action receives an independent
`P(success under the continuation policy)` estimate and the highest is chosen.
Forced actions execute directly. Evaluation terminates on solve, deadlock,
repeated canonical state, no action, or the configured decision limit.

The report includes solve rate, outcome counts, mean push decisions, and mean
primitive movement steps.

When a collector is configured, validation and test also run fresh stochastic
episodes and report `outcome_calibration`. This measures the binary per-action
success questions used by the controller, separately from offline
solver-action agreement under `calibration`.

## Interpretation

The two views answer different questions:

- calibration asks whether the model's probabilities mean what the target
  evidence says they mean;
- environment play asks whether repeated greedy decisions compose into a
  successful trajectory.

A model can be calibrated but weak, or accurate but overconfident. It can also
select good isolated actions while failing long-horizon play. Report both
views, dataset provenance, hardware, seed, and confidence intervals across
multiple seeds for research claims.

## Sokoban evidence targets

The offline dataset labels the push taken by one successful solver path. Its
probabilities therefore concern solver-action agreement. They must not be
described as calibrated probabilities of solving the full level.

The episodic collector adds policy-conditional outcome evidence for selected
actions. It does not prove intrinsic solvability of every action; reports must
retain collector temperature, epsilon, horizon, and policy version.
