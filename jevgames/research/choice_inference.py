"""Frozen pretrained CHOICE inference with shared Sokoban controller rules.

Uses Laya's public predict API and published head/temperatures. No fitting,
custom scoring head, oracle features, deadlock filter or search at inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from statistics import fmean, median
import subprocess
import time
import tomllib

from jevgames.research.bounded_sokoban import exact_graph, state_key, successor
from jevgames.research.objective_ablation import save_json
from sokoban_laya.core import Board
from sokoban_laya.generation import Difficulty, generate_level
from sokoban_laya.laya_policy import BOARD_LEGEND
from sokoban_laya.macros import legal_push_macros
from sokoban_laya.solver import replay


INSTRUCTION = (
    "Solve Sokoban: put every box on a goal. Choose the next reachable push. "
    "Each option gives its box's row r, column c, and push direction. "
    "Coordinates start at 1 in the top-left. Walking is automatic. "
    "Avoid trapping boxes or undoing recent pushes."
)


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def option_description(macro):
    return f"r{macro.box[0] + 1}c{macro.box[1] + 1} {macro.direction}"


def prepare_request(board, history, remaining, presentation, seed, level_id, step):
    macros = list(legal_push_macros(board))
    digest = hashlib.sha256(f"{seed}:{level_id}:{step}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(macros)
    if len(macros) > 26:
        raise ValueError("letter-index study supports at most 26 legal pushes")
    indexed = {chr(65 + i): macro for i, macro in enumerate(macros)}
    state = {
        "board": board.render(), "legend": BOARD_LEGEND,
        "pushes_left": remaining, "recent_pushes": history[-4:],
    }
    if presentation == "structured":
        state.update({
            "player_rc": [board.player[0] + 1, board.player[1] + 1],
            "boxes_rc": [[r + 1, c + 1] for r, c in sorted(board.boxes)],
            "goals_rc": [[r + 1, c + 1] for r, c in sorted(board.goals)],
        })
    elif presentation != "ascii":
        raise ValueError(f"unknown presentation {presentation}")
    question = {
        "type": "choice", "instructions": INSTRUCTION,
        "criteria": {key: option_description(macro) for key, macro in indexed.items()},
    }
    return state, question, indexed


def validate_answer(answer, keys):
    probabilities = answer.get("probabilities", {})
    if set(probabilities) != set(keys) or answer.get("choice") not in keys:
        raise ValueError("model output does not match current legal options")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in probabilities.values()):
        raise ValueError("invalid option probabilities")
    if abs(sum(probabilities.values()) - 1) > .002:
        raise ValueError("option probabilities do not sum to one")
    if probabilities[answer["choice"]] < max(probabilities.values()) - .0002:
        raise ValueError("selected action is not a maximum-probability option")


def audit_native_encoding(agent, state, external_question):
    """Reject silent truncation of instructions, options OR state in native Laya."""
    from laya.common import build_sequence, render_options, serialize_state
    tokenizer = agent.tok
    internal = agent._to_internal(external_question)
    options = render_options(internal)
    ids = [tokenizer.cls_token_id]
    ids += tokenizer(
        f"{internal['t']} question: {internal['ins']}", add_special_tokens=False,
    )["input_ids"]
    ids.append(tokenizer.sep_token_id)
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.append(tokenizer.mask_token_id)
        ids += tokenizer(" " + option, add_special_tokens=False)["input_ids"]
    ids.append(tokenizer.sep_token_id)
    head_length = len(ids)
    ids += tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
    ids.append(tokenizer.sep_token_id)
    actual_ids, actual_markers = build_sequence(
        tokenizer, state, internal, agent.cfg["max_len"], agent.cfg["head_max_len"],
    )
    if ids != actual_ids or markers != actual_markers:
        raise ValueError(
            f"native Laya would truncate input (full={len(ids)}, head={head_length}, "
            f"max_len={agent.cfg['max_len']}, head_max_len={agent.cfg['head_max_len']})"
        )
    return len(ids)


class NativeLaya:
    def __init__(self, config):
        import laya
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(config["source"], revision=config["revision"])
        self.agent = laya.load(snapshot, device=config["device"])
        self.metadata = {
            "source": config["source"], "revision": Path(snapshot).name,
            "api": "laya.load(...).predict", "model_config": self.agent.cfg,
            "laya_version": importlib.metadata.version("laya"),
        }

    def choose(self, state, question):
        tokens = audit_native_encoding(self.agent, state, question)
        started = time.perf_counter()
        result = self.agent.predict(state, {"next_push": question})
        answer = result["answers"]["next_push"]
        validate_answer(answer, question["criteria"])
        return answer, {"seconds": time.perf_counter() - started, "tokens": tokens}


def load_previous_layouts(paths):
    used = set()
    hashes = {}
    for name in paths:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"exclusion manifest unavailable: {path}")
        data = json.loads(path.read_text())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for split in data["splits"].values():
            used.update(split.get("layout_ids", []))
            used.update(row["layout_id"] for row in split.get("records", []))
    return used, hashes


def prepare(config):
    root = Path(config["study"]["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    used, exclusion_hashes = load_previous_layouts(config["dataset"]["exclude_manifests"])
    seed = config["dataset"]["seed"]
    levels = []
    strata = [
        Difficulty("tiny_1box", 6, 6, 1, 2, 5, .08),
        Difficulty("tiny_2box", 6, 6, 2, 3, 7, .08),
        Difficulty("easy_2box", 7, 7, 2, 4, 8, .05),
        Difficulty("medium_3box", 8, 8, 3, 8, 12, .08),
    ]
    for difficulty in strata:
        count = 0
        for _attempt in range(1000):
            seed += 1
            try:
                generated = generate_level(difficulty, seed)
            except RuntimeError:
                continue
            if generated.layout_id in used:
                continue
            if not replay(generated.board, generated.solution).is_solved():
                raise AssertionError("generated witness does not solve the board")
            # Exact labels are only diagnostic metadata; they never enter prepare_request.
            distance = None
            costs = None
            if difficulty.name.startswith("tiny"):
                try:
                    graph = exact_graph(generated.board)
                except OverflowError:
                    continue
                key = state_key(generated.board)
                distance = graph.distances.get(key)
                if distance is None or distance < 2:
                    continue
                costs = {action: None if child not in graph.distances else graph.distances[child] + 1
                         for action, child in graph.edges[key]}
            used.add(generated.layout_id)
            levels.append({
                "layout_id": generated.layout_id, "difficulty": difficulty.name,
                "board": generated.board.render(), "seed": seed,
                "witness": [a.value for a in generated.solution],
                "witness_pushes": generated.reverse_pushes,
                "optimal_pushes": distance, "initial_action_costs": costs,
            })
            count += 1
            if count == config["dataset"]["levels_per_stratum"]:
                break
        if count != config["dataset"]["levels_per_stratum"]:
            raise RuntimeError(f"could not generate enough levels for {difficulty.name}")
    save_json(root / "levels.json", levels)
    save_json(root / "protocol.json", config)
    manifest = {
        "levels": len(levels), "strata": dict(Counter(r["difficulty"] for r in levels)),
        "levels_sha256": hashlib.sha256((root / "levels.json").read_bytes()).hexdigest(),
        "exclusion_manifest_hashes": exclusion_hashes,
        "layout_ids": [row["layout_id"] for row in levels],
    }
    save_json(root / "manifest.json", manifest)
    emit({"phase": "prepared", **manifest})


def trial(levels, policy, method, presentation, seed, horizon):
    started = time.perf_counter()
    results = []
    for level in levels:
        board = Board.from_ascii(level["board"])
        history = []
        trace = []
        seen = set()
        reason = "push_budget"
        rng = random.Random(f"{seed}:{level['layout_id']}")
        revisits = 0
        for step in range(horizon):
            if board.is_solved():
                reason = "solved"
                break
            key = state_key(board)
            revisits += int(key in seen)
            seen.add(key)
            state, question, indexed = prepare_request(
                board, history, horizon - step, presentation, seed, level["layout_id"], step,
            )
            if not indexed:
                reason = "no_action"
                break
            metadata = {"seconds": 0, "tokens": 0}
            if len(indexed) == 1:
                chosen = next(iter(indexed))
                answer = {"choice": chosen, "probabilities": {chosen: 1.0}, "forced": True}
            elif policy is None:
                chosen = rng.choice(list(indexed))
                answer = {"choice": chosen, "probabilities": {k: 1 / len(indexed) for k in indexed}}
            else:
                answer, metadata = policy.choose(state, question)
                chosen = answer["choice"]
            macro = indexed[chosen]
            trace.append({
                "step": step, "observation": state, "criteria": question["criteria"],
                "answer": answer, "selected_macro": macro.label, **metadata,
            })
            history.append(option_description(macro))
            board = successor(board, macro)
            if board.is_solved():
                reason = "solved"
                break
            if board.has_static_deadlock():
                reason = "static_deadlock"
                break
        costs = level["initial_action_costs"]
        first_cost = costs.get(trace[0]["selected_macro"]) if costs and trace else None
        results.append({
            "layout_id": level["layout_id"], "difficulty": level["difficulty"],
            "outcome": reason, "pushes": len(trace), "revisits": revisits,
            "first_action_optimal": first_cost == level["optimal_pushes"] if costs and trace else None,
            "trace": trace, "final_board": board.render(),
        })
        if len(results) % 8 == 0:
            emit({"phase": "progress", "method": method, "presentation": presentation,
                  "seed": seed, "levels": len(results), "solved": sum(r["outcome"] == "solved" for r in results)})
    timing = [t["seconds"] for r in results for t in r["trace"] if t["tokens"]]
    return {
        "method": method, "presentation": presentation, "ordering_seed": seed,
        "solved": sum(r["outcome"] == "solved" for r in results),
        "solve_rate": fmean(r["outcome"] == "solved" for r in results),
        "outcomes": dict(Counter(r["outcome"] for r in results)),
        "seconds": time.perf_counter() - started, "model_calls": len(timing),
        "median_model_ms": 1000 * median(timing) if timing else None,
        "max_tokens": max((t["tokens"] for r in results for t in r["trace"]), default=0),
        "by_difficulty": {d: {"levels": sum(r["difficulty"] == d for r in results),
                               "solved": sum(r["difficulty"] == d and r["outcome"] == "solved" for r in results)}
                          for d in sorted({r["difficulty"] for r in results})},
        "results": results,
    }


def run(config):
    import torch
    root = Path(config["study"]["output_dir"])
    if json.loads((root / "protocol.json").read_text()) != config:
        raise ValueError("protocol changed after registration")
    manifest = json.loads((root / "manifest.json").read_text())
    if hashlib.sha256((root / "levels.json").read_bytes()).hexdigest() != manifest["levels_sha256"]:
        raise ValueError("dataset hash mismatch")
    if (root / "summary.json").exists() or list(root.glob("trial-*.json")):
        raise FileExistsError("study already has results; refusing overwrite")
    levels = json.loads((root / "levels.json").read_text())
    policy = NativeLaya(config["model"])
    sanity = policy.agent.predict(
        {"customer_message": "I was charged twice for the same invoice. Please refund the duplicate payment."},
        {"department": {"type": "choice", "instructions": "Select the department that should handle this request.",
                        "criteria": {"billing": "Payments and refunds", "technical": "Software crashes", "shipping": "Package delivery"}}},
    )
    validate_answer(sanity["answers"]["department"], ("billing", "technical", "shipping"))
    emit({"phase": "native_sanity", "answer": sanity["answers"]["department"]})
    save_json(root / "runtime.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": policy.metadata, "sanity": sanity,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    })
    results = []
    for presentation in config["study"]["presentations"]:
        for seed in config["study"]["ordering_seeds"]:
            for method, active in (("random", None), ("laya_pretrained", policy)):
                report = trial(levels, active, method, presentation, seed, config["study"]["max_pushes"])
                if torch.cuda.is_available():
                    report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                save_json(root / f"trial-{method}-{presentation}-{seed}.json", report)
                results.append({k: v for k, v in report.items() if k != "results"})
                emit({"phase": "trial_complete", **results[-1]})
    summary = {"config": config, "manifest": manifest, "trials": results,
               "jev_status": "not_run_user_selected_laya_first"}
    save_json(root / "summary.json", summary)
    emit({"phase": "study_complete", "trials": len(results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("config")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    (prepare if args.command == "prepare" else run)(config)


if __name__ == "__main__":
    main()
