# Sokoban case study

Sokoban demonstrates dynamic textual choices, irreversible failures, sparse
long-horizon success, procedural evidence generation, and the difference
between calibrated local decisions and complete environment solving.

## Decision space

The model chooses reachable box pushes such as:

```text
box_r2_c6_push_left
box_r3_c4_push_down
```

A deterministic BFS controller executes the walking path required for the
chosen push. This removes navigation-only choices while preserving strategic
box movement.

## Current evidence

The reverse-play generator starts with boxes on goals, applies legal reverse
pulls, and records a solution back to the generated initial board. Primitive
paths are compressed into one evidence row per push.

Current pilot schedule:

| Difficulty | Board | Boxes | Reverse pushes | Share |
|---|---:|---:|---:|---:|
| `easy_2box` | 7×7 | 2 | 4–8 | 40% |
| `medium_3box` | 8×8 | 3 | 8–14 | 35% |
| `hard_4box` | 10×10 | 4 | 12–20 | 25% |

The `sokoban_solver` evidence plugin turns the observed solver push into a
one-hot target among all currently legal push macros. Evidence provenance is
`solver_solution`.

## What RLCD learns here

The current target asks which legal push matches one successful solver path.
RLCD trains the probability distribution over that event. It does not directly
train a calibrated probability that each push will eventually solve the level.

This distinction matters when several pushes are valid, when a successful
continuation depends on later policy quality, or when the solver returns only
one of several solutions.

## Next evidence design

Outcome-calibrated Sokoban evidence should:

1. enumerate legal push macros at representative states;
2. evaluate each push under a fixed continuation policy or bounded solver;
3. repeat stochastic continuations where applicable;
4. retain success, failure, and unresolved counts separately;
5. produce a target distribution whose event meaning is written into the
   manifest;
6. prevent states or layouts from crossing data splits.

This evidence can enter the same RLCD strategy as soft targets. The framework
does not need an episode-reward GRPO phase to consume it.

## Environment evaluation

After training, the model plays each held-out level greedily with no search
fallback. A canonical state combines box positions with the player's reachable
region. Runs stop on solve, deadlock, repeat, no legal push, or push limit.

Solve rate tests sequential competence. Calibration metrics test probability
quality against the evidence event. Neither metric substitutes for the other.

## Known limits

- static deadlock detection is conservative;
- reverse-generated levels differ from handcrafted distributions;
- one solver path does not identify all acceptable actions;
- the Sokoban task currently emits only `choice` questions, while the generic
  model contract also supports `score` and `noul`;
- hard handcrafted Sokoban may require explicit planning or policy-guided
  search even with a strong calibrated local decision model.
