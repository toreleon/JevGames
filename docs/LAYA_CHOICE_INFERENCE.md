# Pretrained Laya CHOICE experiment — 2026-09-22

This study tests an inference architecture inspired by the browser-use
`jev-ultrafast` agent: an existing decision model sees the current state,
recent actions and indexed executable options, then directly selects an action.
The user selected **Laya first**; this is not a test of TypeSafe's hosted Jev.

Completed results: [LAYA_CHOICE_INFERENCE_RESULTS.md](LAYA_CHOICE_INFERENCE_RESULTS.md).

## Frozen protocol

- Published `convaiinnovations/laya` revision
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, loaded through the native
  `laya.load(...).predict` API, including its trained decision head and native
  temperature tables. No fine-tuning, new random head, calibration fitting or
  test-driven prompt selection.
- 32 fresh procedural layouts: eight each of tiny one-box, tiny two-box, easy
  two-box and medium three-box. All have replay-verified solutions. Exact
  minimum push distances are computed for tiny levels as diagnostic metadata.
  These distances, witness trajectories and solver information never enter
  the policy observation.
- Exclude every layout from both previous pilot manifests and the paired
  objective study. The dataset and exclusion manifests have SHA-256 receipts.
- Every reachable push is offered, including pushes causing static deadlock.
  There is no search, deadlock mask, novelty bonus or solver fallback. Forced
  actions execute without a model call; zero options terminate the episode.
- Stop on solved, static deadlock, no action or 16 pushes. Revisited states are
  recorded but not automatically pruned. Independent board rules determine
  success rather than a model's self-reported completion.
- Shared instruction: all boxes onto goals, one-based row/column coordinates,
  automatic walking, avoid trapped boxes and undoing recent pushes.
- Two presentations: ASCII board/legend/history/budget, and that identical
  state augmented with explicit player, box and goal coordinates. Both receive
  the same indexed action descriptions and last four pushes.
- Three option-order seeds (17, 29, 43), paired across presentations and methods.
  These are **presentation replicates**, not training seeds or 96 independent
  boards. Laya uses its native greedy selected choice; random uniformly samples
  the same action set. Ordering is stable for each level/step/seed.
- Each native call reconstructs the full untruncated token sequence and checks
  it against Laya's actual `build_sequence`. Instruction, option and state
  truncation abort the run rather than silently measuring a damaged prompt.
  Native 512-token total/192-token head limits are preserved.
- A simple billing-routing query checks that the published head can perform a
  basic typed decision. It does not select the game prompt or change the study.

## Endpoints and interpretation

Primary endpoint: solved count by difficulty and mean across option orders,
compared with the paired random controller. Also report initial optimal-action
selection where exact labels are available, repeat/deadlock rates, option-position
bias, input token maxima, native-call latency, GPU memory and complete traces.
CHOICE probabilities are a policy distribution, not a calibrated probability
that an episode will be solved; no such calibration claim is made.

This probes whether pretrained typed-decision capability transfers to Sokoban
under this representation and budget. A failed trial cannot establish that all
Laya models, TypeSafe Jev, language models, or learned game policies cannot work.
The experiment includes only generated levels up to three boxes, not a hard
handcrafted Sokoban benchmark.

## Reproduction

```bash
uv sync --locked --extra laya --extra train --python 3.12
uv run python -m jevgames.research.choice_inference prepare configs/sokoban_laya_choice_inference.toml
uv run python -m jevgames.research.choice_inference run configs/sokoban_laya_choice_inference.toml
```

Run on the RTX server with the committed CUDA config. Existing output is never
overwritten; generated levels, native metadata, per-action traces and summary
live under `runs/laya-choice-inference-20260922/`. No paid model API is called.

Reference architecture:
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast).
