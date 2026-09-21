# Sokoban case study

Sokoban is the first task plugin for Jev Games. It demonstrates dynamic textual
actions, irreversible failures, sparse terminal reward, expert trajectory
compression, procedural environment generation, and model-only evaluation.

Sokoban is not embedded in the generic engine. The plugin lives in
`jevgames/tasks/sokoban.py` and reuses the implementation package
`sokoban_laya`.

## State encoding

Boards use standard ASCII symbols:

| Symbol | Meaning |
|---|---|
| `#` | wall |
| space | floor |
| `.` | goal |
| `$` | box |
| `*` | box on goal |
| `@` | player |
| `+` | player on goal |

Standard XSB files may use exterior whitespace. `Board.from_xsb` flood-fills
boundary whitespace and converts it to non-playable walls.

## Why push macros

A primitive policy chooses among walking and pushing directions. This creates
long horizons dominated by navigation and easily learns two-step walking loops.

The plugin instead presents reachable pushes:

```text
box_r2_c6_push_left
box_r3_c4_push_down
```

For each option, BFS verifies that the player can reach the required side of
the box without moving another box. After the model selects a push, the
environment executes the shortest walking path and one push.

This transforms a primitive solution into strategic decisions. In the measured
Microban 3 example:

- primitive trajectory: 41 actions;
- push-macro trajectory: 13 decisions.

## Expert generation

Two sources are supported:

1. A* solutions for small supplied levels;
2. reverse-play generation for scalable curricula.

The reverse generator begins with every box on a goal, applies legal reverse
pulls, records inverse actions, and verifies that replay solves the resulting
board. It therefore produces a known solution without running A* on every
generated instance.

Current pilot schedule:

| Difficulty | Board | Boxes | Reverse pushes | Share |
|---|---:|---:|---:|---:|
| `easy_2box` | 7×7 | 2 | 4–8 | 40% |
| `medium_3box` | 8×8 | 3 | 8–14 | 35% |
| `hard_4box` | 10×10 | 4 | 12–20 | 25% |

Internal walls are generated probabilistically, then disconnected floor is
closed. The manifest records seed, difficulty, layout hash, level hash, and
expert lengths.

## Reward

Default task reward:

```text
+50.0  solved
 +4.0  box newly on a goal
 -4.0  box removed from a goal
 -0.02 each push
 -0.002 each walking step inside the push macro
-15.0 static deadlock or no reachable push
 -4.0 repeated canonical state
 -1.0 decision limit
```

The canonical state includes box positions and the player's reachable region,
not the exact player cell. Positions connected without moving a box are
strategically equivalent under the macro action space.

## Dataset generation

```bash
uv run python scripts/generate_curriculum.py \
  --output data/pilot \
  --train 2000 \
  --validation 200 \
  --test 200 \
  --seed 20260921
```

Measured output:

- 2,400 unique layouts generated in 6.78 seconds;
- 102,404 primitive train decisions;
- 20,362 macro train decisions;
- approximately 25 MB train JSONL;
- disjoint layout hashes across train, validation, and test.

## Measured experiments

### Microban 3

A task-specific run reached 88.89% macro expert accuracy and produced a greedy
policy that solved the training level in 13 pushes. Because the level was used
for expert training, this proves pipeline learning and execution, not held-out
generalization.

### 100-level pilot benchmark

One warm-up epoch and one online iteration produced solved training episodes
but 0/20 held-out validation solves. This is expected for a bounded mechanics
benchmark and is evidence that training reward must not replace validation.

### XSokoban Round 1

The small trained policies did not solve the six-box level and entered a static
deadlock. A reactive one-step policy may ultimately require policy-guided beam
search or another planning component for robust hard-level performance.

## Known limitations

- Static deadlock detection is conservative and does not cover every dynamic
  pattern.
- Reverse-generated levels are solvable but do not match every handcrafted
  Sokoban distribution.
- Current expert data records primitive actions before compression.
- The task plugin does not yet expose per-difficulty metrics through the generic
  benchmark contract; the compatibility benchmark does.
- Generalization to handcrafted hard levels remains unproven.

## Recommended research progression

1. train the 2,000-level pilot and select checkpoints on validation solve rate;
2. inspect solve rate separately for two-, three-, and four-box levels;
3. add stronger dynamic-deadlock signals;
4. scale to 10,000 unique layouts before increasing repeated rollouts heavily;
5. compare frozen versus partially unfrozen Laya backbones;
6. add policy-guided search as a separately reported hybrid benchmark.
