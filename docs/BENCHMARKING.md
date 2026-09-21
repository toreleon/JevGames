# Benchmarking protocol

Jev Games evaluates two properties separately: whether reported probabilities
are calibrated and whether the composed policy solves complete environments.

## Split discipline

Train, validation, and test must be disjoint by underlying environment layout,
not merely by serialized state. Sokoban manifests record layout and level
hashes so trajectories from one layout cannot cross splits.

Training reads only `data.train`. Validation and test are loaded after the
checkpoint is trained.

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
| `mean_confidence` | mean maximum reported probability |
| `mean_proper_score` | adapter-defined strictly proper score |
| `reliability_bins` | count, confidence, accuracy, and gap per occupied bin |

ECE depends on binning and sample count. Always inspect reliability bins and
NLL/Brier alongside it.

## Environment benchmark

The saved model plays greedily with no solver fallback and no weight updates.
Forced actions execute directly; states with multiple legal pushes are batched
through the model. Evaluation terminates on solve, deadlock, repeated canonical
state, no action, or the configured decision limit.

The report includes solve rate, outcome counts, mean push decisions, and mean
primitive movement steps.

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

## Current Sokoban target

The current dataset labels the push taken by one successful solver path. Its
probabilities therefore concern solver-action agreement. They must not be
described as calibrated probabilities of solving the full level.

An outcome-calibrated successor dataset should evaluate each legal push across
controlled continuations, preserve unresolved attempts as unknown, and record
the sampling policy and budget in its manifest.
