# Why a proper reward can yield a miscalibrated update

This is a **post-hoc** diagnostic added while the frozen paired Sokoban study
was running. It does not change its data, objective, hyperparameters or budget.

## Constant-input experiment

Every example has the same input. There are exactly 70 false and 30 true
outcomes. A model with two shared trainable logits can represent the correct
conditional probability, p(true)=0.3. There is no text model, spatial reasoning,
label error or hidden policy-continuation assumption.

All controls run 128 AdamW steps with LR 0.03, no weight decay, and the same
log+spherical proper reward used in the framework. Report-sampling controls
use G=16 and sigma annealed from 0.6 to 0.15. Seeds: 17, 29, 43.

| Method | Predicted p(true), seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| Current RLCD, per-example standardized advantage | 0.0000824 | 0.0000848 | 0.0000847 |
| Direct proper-score gradient | 0.299959 | 0.299959 | 0.299959 |
| Gaussian score-function control, leave-one-out baseline without std division | 0.294421 | 0.298845 | 0.299760 |

The normalized variant's expected proper score is approximately -2.47; the
other two are approximately -0.2301, near the truthful forecast optimum.
The score-function control is a diagnostic, not yet a new production strategy.

## Interpretation

For a small perturbation of a binary decision's logit difference, write the
reward as `r_y(z + epsilon) ≈ r_y(z) + r'_y(z) * epsilon`. Subtracting its group
mean and dividing by its within-example standard deviation approximately
removes the derivative magnitude:

`A_y ≈ sign(r'_y(z)) * epsilon / std(epsilon)`.

The Gaussian score-function update thus depends mainly on the *sign* of the
label's desired change. With 70% false outcomes, their updates outvote the 30%
true outcomes even when the model already reports p(true)=0.3. This encourages
the majority class rather than the truthful probability. The approximation
explains the observed toy failure; it is not a proof about every parameter
setting or every algorithm called RLCD.

A baseline independent of the sampled report can reduce variance without this
label-dependent rescaling. The leave-one-out control retains reward magnitude.
Gaussian smoothing still changes the optimized score relative to reporting
the unperturbed probabilities, so it warrants separate evaluation.

These findings correct the earlier inference that aligned local logit gradient
directions or a high overfit accuracy established the estimator's adequacy.
Those checks did not test whether conflicting outcomes for the same input
converge to their true frequency. Likewise, a proper reward's theoretical
optimum is not automatically preserved by an arbitrary normalized update.

## Reproduction

```bash
uv run --extra train python scripts/diagnose_probability_objective.py \
  --output runs/sokoban-objective-ablation-20260921/bernoulli_control.json
```

The result applies to **our current group-standardized Gaussian-report
implementation**. It cannot establish limitations of TypeSafe's proprietary
training method or of all probability-trained language models.
