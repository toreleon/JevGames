"""Exact push-distance oracle and horizon-labelled research fixtures.

The complete reachable macro graph is built per layout, with only provably
losing static-deadlock states pruned. Reverse BFS from every solved state gives
exact minimum push counts. A graph that exceeds its state budget is discarded;
budget exhaustion must never turn into a negative label.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from jevgames.contracts import DecisionKind, DecisionOption, DecisionQuestion
from sokoban_laya.core import Board
from sokoban_laya.generation import Difficulty, generate_level
from sokoban_laya.laya_policy import BOARD_LEGEND
from sokoban_laya.macros import legal_push_macros, reachable_walk_paths


def state_key(board: Board):
    return board.boxes, min(reachable_walk_paths(board))


def successor(board: Board, macro) -> Board:
    # macro is already executable, so after its one push the player occupies
    # the old box square; the preceding walking path is irrelevant to push cost.
    delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[macro.direction]
    destination = (macro.box[0] + delta[0], macro.box[1] + delta[1])
    return Board(
        board.width, board.height, board.walls, board.goals,
        (board.boxes - {macro.box}) | {destination}, macro.box,
    )


@dataclass
class PushGraph:
    boards: dict
    edges: dict
    distances: dict


def exact_graph(initial: Board, max_states: int = 2500) -> PushGraph:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    root = state_key(initial)
    boards = {root: initial}
    edges = {}
    parents = {}
    frontier = deque([root])
    goals = []
    dead_squares = initial.static_dead_squares()
    while frontier:
        key = frontier.popleft()
        board = boards[key]
        edges[key] = []
        if board.is_solved():
            goals.append(key)
            continue
        if any(box in dead_squares and box not in board.goals for box in board.boxes):
            continue
        for macro in legal_push_macros(board):
            child = successor(board, macro)
            child_key = state_key(child)
            if child_key not in boards:
                if len(boards) >= max_states:
                    raise OverflowError("complete push graph exceeds state budget; labels unknown")
                boards[child_key] = child
                frontier.append(child_key)
            edges[key].append((macro.label, child_key))
            parents.setdefault(child_key, []).append(key)
    distances = {key: 0 for key in goals}
    frontier = deque(goals)
    while frontier:
        child = frontier.popleft()
        for parent in parents.get(child, []):
            if parent not in distances:
                distances[parent] = distances[child] + 1
                frontier.append(parent)
    return PushGraph(boards, edges, distances)


def question(description: str, horizon: int) -> DecisionQuestion:
    if horizon < 1:
        raise ValueError("horizon includes the first push and must be positive")
    return DecisionQuestion(
        key="exists_solution_within_push_budget",
        kind=DecisionKind.NOUL,
        instruction=(
            f"First action: {description}. Including this first push, does ANY legal "
            f"sequence solve the board within {horizon} total box pushes? "
            "Walking costs zero pushes. The player may choose the best continuation."
        ),
        options=(DecisionOption("false", "No such solution exists"),
                 DecisionOption("true", "Such a solution exists")),
    )


def observation(board: str) -> dict:
    return {"board": board, "legend": BOARD_LEGEND}


def encode_record(adapter, row: dict):
    label = int(row["label"])
    return adapter.encode(
        observation(row["board"]), question(row["description"], row["horizon"]),
        (1.0 - label, float(label)),
    )


def generate_dataset(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(config["seed"])
    used_layouts = set()
    excluded_layouts = set()
    for manifest_path in config.get("exclude_manifests", []):
        path = Path(manifest_path)
        if path.exists():
            old = json.loads(path.read_text())
            excluded_layouts.update(
                row["layout_id"] for split in old["splits"].values() for row in split["records"]
            )
    used_layouts.update(excluded_layouts)
    manifest = {"config": config, "splits": {}, "excluded_layouts": len(excluded_layouts)}
    candidate_seed = config["seed"]
    for split in ("train", "validation", "test"):
        levels = []
        evidence = []
        rejected = Counter()
        attempts = 0
        while len(levels) < config[f"{split}_levels"]:
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("failed to obtain enough complete, disjoint layouts")
            candidate_seed += 1
            boxes = 1 + len(levels) % 2
            difficulty = Difficulty(f"tiny_{boxes}box", 6, 6, boxes, 2, 7, 0.08)
            try:
                generated = generate_level(difficulty, candidate_seed)
            except RuntimeError:
                rejected["generation"] += 1
                continue
            if generated.layout_id in used_layouts:
                rejected["duplicate_layout"] += 1
                continue
            try:
                graph = exact_graph(generated.board, config["max_graph_states"])
            except OverflowError:
                rejected["graph_budget"] += 1
                continue
            root = state_key(generated.board)
            distance = graph.distances.get(root)
            if distance is None or not 2 <= distance <= config["horizon"]:
                rejected["root_distance"] += 1
                continue
            used_layouts.add(generated.layout_id)
            levels.append({
                "layout_id": generated.layout_id, "board": generated.board.render(),
                "seed": candidate_seed, "boxes": boxes, "distance": distance,
                "graph_states": len(graph.boards),
            })
            states = [key for key in graph.boards if 1 <= graph.distances.get(key, 0) <= config["horizon"]]
            rng.shuffle(states)
            states = [root] + [key for key in states if key != root][:config["states_per_level"] - 1]
            for state_index, key in enumerate(states):
                board = graph.boards[key]
                macros = {macro.label: macro for macro in legal_push_macros(board)}
                horizons = {max(1, graph.distances[key] - 1), graph.distances[key],
                            min(config["horizon"], graph.distances[key] + 1)}
                for horizon in sorted(horizons):
                    group = f"{generated.layout_id}:{state_index}:{horizon}"
                    for action, child in graph.edges[key]:
                        child_distance = graph.distances.get(child)
                        action_cost = None if child_distance is None else child_distance + 1
                        evidence.append({
                            "layout_id": generated.layout_id, "group": group,
                            "boxes": boxes, "board": board.render(), "action": action,
                            "description": macros[action].description, "horizon": horizon,
                            "label": action_cost is not None and action_cost <= horizon,
                            "oracle_push_cost": action_cost, "state_distance": graph.distances[key],
                        })
        # Only train is subsampled; validation/test retain every action in every group.
        if split == "train":
            rng.shuffle(evidence)
            evidence = evidence[:config["train_evidence"]]
            if len(evidence) != config["train_evidence"]:
                raise ValueError("insufficient evidence for the registered training budget")
        payload = {"levels": levels, "evidence": evidence}
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        (output / f"{split}.json").write_bytes(encoded)
        manifest["splits"][split] = {
            "sha256": hashlib.sha256(encoded).hexdigest(), "levels": len(levels),
            "evidence": len(evidence), "labels": dict(Counter(int(r["label"]) for r in evidence)),
            "layout_ids": [r["layout_id"] for r in levels], "rejected": dict(rejected),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
