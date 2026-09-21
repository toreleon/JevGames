# Pretrained Laya CHOICE results — 2026-09-22

The frozen pretrained Laya policy did not show a solve-rate advantage over uniform random actions in this experiment. ASCII matched random in aggregate; adding explicit coordinates did not improve it. This is a transfer test, not a training experiment or a test of TypeSafe Jev.

Protocol: [LAYA_CHOICE_INFERENCE.md](LAYA_CHOICE_INFERENCE.md). 32 fresh generated layouts × three option-order replicates, with a 16-push cap. The same 32 layouts recur in every arm: 96 episodes are not 96 independent puzzles. No fitting, search, deadlock filtering or oracle observation was used.

## Solves

Each entry is solved/32 for option-order seeds 17, 29 and 43.

| Policy | Solved per order | Mean solve rate |
|---|---|---:|
| random/ascii | [2, 4, 4] | 10.42% |
| laya_pretrained/ascii | [3, 2, 5] | 10.42% |
| laya_pretrained/structured | [2, 2, 2] | 6.25% |

Random was also run with structured input: every action and outcome matched its ASCII run, as expected. Do not count those duplicates as extra random evidence.

| Difficulty (8 layouts each) | Random | Laya ASCII | Laya structured |
|---|---|---|---|
| tiny_1box | [1, 1, 3] | [3, 1, 2] | [2, 1, 1] |
| tiny_2box | [0, 1, 1] | [0, 1, 2] | [0, 1, 0] |
| easy_2box | [0, 1, 0] | [0, 0, 1] | [0, 0, 1] |
| medium_3box | [1, 1, 0] | [0, 0, 0] | [0, 0, 0] |

## Behavior diagnostics

The first-action oracle is available only for the 16 tiny layouts. These labels were never shown to the model. Position frequencies below exclude forced actions; the uniform reference is computed over each policy's own visited states, not a matched-state causal comparison.

| Policy | Optimal first push | Static-deadlock episodes | No-action episodes | Revisited-state episodes |
|---|---:|---:|---:|---:|
| random/ascii | 21/48 | 79/96 | 5/96 | 20/96 |
| laya_pretrained/ascii | 28/48 | 72/96 | 13/96 | 22/96 |
| laya_pretrained/structured | 28/48 | 80/96 | 9/96 | 22/96 |

| Policy | First option selected | Uniform expectation | Same initial action across all orders |
|---|---:|---:|---:|
| random/ascii | 68/320 | 69.2/320 | 2/31 |
| laya_pretrained/ascii | 149/341 | 79.5/341 | 10/31 |
| laya_pretrained/structured | 131/338 | 78.5/338 | 11/31 |

## Runtime and verification

- Runtime: `NVIDIA GeForce RTX 5060 Ti`, PyTorch `2.14.0+cu130`, native Laya `0.3.4`; code `b14148b99aecf6b749fd9e943dda82d118d1ab1e`.
- Published model snapshot: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Original trained head, BF16 autocast and temperature tables retained.
- Non-game sanity query: selected `billing` with probability 0.9786 for a duplicate-payment refund request.
- Peak CUDA tensor allocation: 2.300 GiB for the whole process, including the already loaded model during random trials; not random-policy memory or total device VRAM.
- All 384 saved episode traces replayed through the primitive walking/pushing engine, reproducing observations, offered options, chosen actions, final boards and outcomes.
- Every native game request passed an exact full-sequence truncation audit. No request exceeded the native 512-token total/192-token head budgets.

| Presentation | Native game calls | Median call ms | Maximum tokens | Evaluation seconds |
|---|---:|---:|---:|---:|
| ascii | 341 | 17.09 | 276 | 6.69 |
| structured | 338 | 18.40 | 326 | 7.08 |

Timings exclude model download/load and the sanity warm-up. Native call time includes native tokenization/inference/result conversion but excludes the separate pre-call encoding audit; evaluation time includes the controller and that audit.

## Interpretation and next gate

The semantic sanity query works, so this is not evidence that the published head cannot make any useful decision. It does show that a working typed interface and low latency do not establish Sokoban spatial planning. Neither coordinates nor using the pretrained CHOICE head was sufficient under this protocol.

Do not extend training merely from this result: there was no training here. A separate, useful next experiment would establish a learnability gate on tiny boards with game-specific CHOICE supervision, variable action counts and option-order augmentation, then evaluate unseen layouts before scaling episodes. Keep any RLCD objective comparison separate from the representation/learnability check.

This small generated suite does not test hard handcrafted Sokoban, the model's unknown original training corpus, another prompt family, tuned game training, or TypeSafe Jev. Ordering replicates are correlated, and no broad statistical superiority/inferiority claim is justified. A later study needs new layouts rather than tuning on these reported evaluation boards.

## Artifacts

Raw per-step traces and logs: `runs/laya-choice-inference-20260922/` on the local checkout and server (ignored in Git). The adjacent JSON receipt preserves the frozen levels, runtime, aggregates, per-episode outcomes and SHA-256 hashes of every raw JSON artifact. No paid API was used.

```bash
uv run python scripts/summarize_choice_inference.py runs/laya-choice-inference-20260922 --output docs/LAYA_CHOICE_INFERENCE_RESULTS.md
```
