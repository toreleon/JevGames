"""Audit native CHOICE trajectories and produce an aggregate research receipt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean, median

from jevgames.research.choice_inference import option_description, prepare_request, validate_answer
from sokoban_laya.core import Board
from sokoban_laya.macros import apply_push_macro, legal_push_macros


def audit_trial(report, levels, horizon):
    """Re-execute saved choices through the primitive walking/pushing engine."""
    if [r["layout_id"] for r in report["results"]] != list(levels):
        raise ValueError("trial layouts/order do not match the frozen dataset")
    for row in report["results"]:
        level = levels[row["layout_id"]]
        board = Board.from_ascii(level["board"])
        history = []
        if len(row["trace"]) > horizon or row["pushes"] != len(row["trace"]):
            raise ValueError("trace exceeds the push budget")
        reason = "push_budget"
        for step, trace in enumerate(row["trace"]):
            if trace["step"] != step:
                raise ValueError("trace step index mismatch")
            state, question, actions = prepare_request(
                board, history, horizon - step, report["presentation"],
                report["ordering_seed"], row["layout_id"], step,
            )
            if state != trace["observation"] or question["criteria"] != trace["criteria"]:
                raise ValueError("saved state or offered actions mismatch")
            validate_answer(trace["answer"], actions)
            action = actions[trace["answer"]["choice"]]
            if action.label != trace["selected_macro"]:
                raise ValueError("saved selection does not match the model answer")
            board, _ = apply_push_macro(board, action)
            history.append(option_description(action))
            if board.is_solved() or board.has_static_deadlock():
                reason = "solved" if board.is_solved() else "static_deadlock"
                if step != len(row["trace"]) - 1:
                    raise ValueError("trace continues after a terminal state")
        if reason == "push_budget" and len(row["trace"]) < horizon:
            if legal_push_macros(board):
                raise ValueError("trace stops before a terminal state or budget exhaustion")
            reason = "no_action"
        if board.render() != row["final_board"] or reason != row["outcome"]:
            raise ValueError("primitive replay does not reproduce the saved outcome")
        costs = level["initial_action_costs"]
        expected_optimal = (costs.get(row["trace"][0]["selected_macro"]) == level["optimal_pushes"]
                            if costs and row["trace"] else None)
        if row["first_action_optimal"] != expected_optimal:
            raise ValueError("first-action diagnostic does not match frozen oracle metadata")
    if sum(r["outcome"] == "solved" for r in report["results"]) != report["solved"]:
        raise ValueError("solved total does not match episodes")


def aggregate(reports):
    rows = [r for report in reports for r in report["results"]]
    choices = [t for r in rows for t in r["trace"] if len(t["criteria"]) > 1]
    calls = [t for t in choices if t["tokens"]]
    first_labels = [r["first_action_optimal"] for r in rows if r["first_action_optimal"] is not None]
    first_actions = defaultdict(list)
    for row in rows:
        if row["trace"] and len(row["trace"][0]["criteria"]) > 1:
            first_actions[row["layout_id"]].append(row["trace"][0]["selected_macro"])
    return {
        "solved_by_order": [r["solved"] for r in reports],
        "episodes": len(rows), "solve_rate": fmean(r["outcome"] == "solved" for r in rows),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "episodes_with_revisits": sum(r["revisits"] > 0 for r in rows),
        "first_optimal": {"correct": sum(first_labels), "labelled": len(first_labels)},
        "first_action_order_stability": {
            "identical_across_all_orders": sum(len(set(a)) == 1 for a in first_actions.values()),
            "layouts_with_initial_choice": len(first_actions),
        },
        "nonforced_decisions": len(choices),
        "first_option_selected": sum(t["answer"]["choice"] == next(iter(t["criteria"])) for t in choices),
        "uniform_expected_first_option": sum(1 / len(t["criteria"]) for t in choices),
        "model_calls": len(calls),
        "median_model_ms": 1000 * median(t["seconds"] for t in calls) if calls else None,
        "max_tokens": max((t["tokens"] for t in calls), default=0),
        "evaluation_seconds": sum(r["seconds"] for r in reports),
        "by_difficulty": {d: [r["by_difficulty"][d]["solved"] for r in reports]
                          for d in reports[0]["by_difficulty"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    summary = json.loads((root / "summary.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    levels = {r["layout_id"]: r for r in json.loads((root / "levels.json").read_text())}
    if hashlib.sha256((root / "levels.json").read_bytes()).hexdigest() != summary["manifest"]["levels_sha256"]:
        raise ValueError("dataset hash mismatch")
    config = summary["config"]
    groups, reports = {}, {}
    for presentation in config["study"]["presentations"]:
        for method in ("random", "laya_pretrained"):
            trials = []
            for seed in config["study"]["ordering_seeds"]:
                filename = f"trial-{method}-{presentation}-{seed}.json"
                report = json.loads((root / filename).read_text())
                audit_trial(report, levels, config["study"]["max_pushes"])
                reports[filename] = report
                trials.append(report)
            groups[f"{method}/{presentation}"] = aggregate(trials)
    # The representation must not affect the random policy's actual trajectory.
    for seed in config["study"]["ordering_seeds"]:
        left, right = [reports[f"trial-random-{p}-{seed}.json"]["results"] for p in ("ascii", "structured")]
        for a, b in zip(left, right, strict=True):
            if a["outcome"] != b["outcome"] or [t["selected_macro"] for t in a["trace"]] != [t["selected_macro"] for t in b["trace"]]:
                raise ValueError("random trajectories differ across representations")
    counts = {name: group["solved_by_order"] for name, group in groups.items()}
    sanity = runtime["sanity"]["answers"]["department"]
    peak = max(t["peak_cuda_allocated_gib"] for t in summary["trials"])
    lines = [
        "# Pretrained Laya CHOICE results — 2026-09-22", "",
        "The frozen pretrained Laya policy did not show a solve-rate advantage over "
        "uniform random actions in this experiment. ASCII matched random in aggregate; "
        "adding explicit coordinates did not improve it. This is a transfer test, not a "
        "training experiment or a test of TypeSafe Jev.", "",
        "Protocol: [LAYA_CHOICE_INFERENCE.md](LAYA_CHOICE_INFERENCE.md). "
        "32 fresh generated layouts × three option-order replicates, with a 16-push cap. "
        "The same 32 layouts recur in every arm: 96 episodes are not 96 independent puzzles. "
        "No fitting, search, deadlock filtering or oracle observation was used.", "",
        "## Solves", "",
        "Each entry is solved/32 for option-order seeds 17, 29 and 43.", "",
        "| Policy | Solved per order | Mean solve rate |",
        "|---|---|---:|",
    ]
    for name in ("random/ascii", "laya_pretrained/ascii", "laya_pretrained/structured"):
        lines.append(f"| {name} | {counts[name]} | {groups[name]['solve_rate']:.2%} |")
    lines.extend(["", "Random was also run with structured input: every action and outcome matched "
                  "its ASCII run, as expected. Do not count those duplicates as extra random evidence.", "",
                  "| Difficulty (8 layouts each) | Random | Laya ASCII | Laya structured |",
                  "|---|---|---|---|"])
    names = ("random/ascii", "laya_pretrained/ascii", "laya_pretrained/structured")
    for difficulty in ("tiny_1box", "tiny_2box", "easy_2box", "medium_3box"):
        lines.append(f"| {difficulty} | " + " | ".join(str(groups[n]["by_difficulty"][difficulty]) for n in names) + " |")
    lines.extend(["", "## Behavior diagnostics", "",
                  "The first-action oracle is available only for the 16 tiny layouts. "
                  "These labels were never shown to the model. Position frequencies below "
                  "exclude forced actions; the uniform reference is computed over each "
                  "policy's own visited states, not a matched-state causal comparison.", "",
                  "| Policy | Optimal first push | Static-deadlock episodes | No-action episodes | Revisited-state episodes |",
                  "|---|---:|---:|---:|---:|"])
    for name in names:
        g = groups[name]
        lines.append(f"| {name} | {g['first_optimal']['correct']}/{g['first_optimal']['labelled']} | "
                     f"{g['outcomes'].get('static_deadlock', 0)}/96 | {g['outcomes'].get('no_action', 0)}/96 | "
                     f"{g['episodes_with_revisits']}/96 |")
    lines.extend(["", "| Policy | First option selected | Uniform expectation | Same initial action across all orders |",
                  "|---|---:|---:|---:|"])
    for name in names:
        g = groups[name]
        stable = g["first_action_order_stability"]
        lines.append(f"| {name} | {g['first_option_selected']}/{g['nonforced_decisions']} | "
                     f"{g['uniform_expected_first_option']:.1f}/{g['nonforced_decisions']} | "
                     f"{stable['identical_across_all_orders']}/{stable['layouts_with_initial_choice']} |")
    lines.extend(["", "## Runtime and verification", "",
                  f"- Runtime: `{runtime['gpu']}`, PyTorch `{runtime['torch']}`, native Laya "
                  f"`{runtime['model']['laya_version']}`; code `{runtime['git_commit']}`.",
                  f"- Published model snapshot: `{runtime['model']['revision']}`. "
                  "Original trained head, BF16 autocast and temperature tables retained.",
                  f"- Non-game sanity query: selected `{sanity['choice']}` with "
                  f"probability {sanity['probabilities'][sanity['choice']]:.4f} for a duplicate-payment refund request.",
                  f"- Peak CUDA tensor allocation: {peak:.3f} GiB for the whole process, "
                  "including the already loaded model during random trials; not random-policy memory or total device VRAM.",
                  "- All 384 saved episode traces replayed through the primitive walking/pushing "
                  "engine, reproducing observations, offered options, chosen actions, final boards and outcomes.",
                  "- Every native game request passed an exact full-sequence truncation audit. "
                  "No request exceeded the native 512-token total/192-token head budgets.", "",
                  "| Presentation | Native game calls | Median call ms | Maximum tokens | Evaluation seconds |",
                  "|---|---:|---:|---:|---:|"])
    for presentation in ("ascii", "structured"):
        g = groups[f"laya_pretrained/{presentation}"]
        lines.append(f"| {presentation} | {g['model_calls']} | {g['median_model_ms']:.2f} | "
                     f"{g['max_tokens']} | {g['evaluation_seconds']:.2f} |")
    lines.extend(["", "Timings exclude model download/load and the sanity warm-up. Native call time "
                  "includes native tokenization/inference/result conversion but excludes the separate "
                  "pre-call encoding audit; evaluation time includes the controller and that audit.", "",
                  "## Interpretation and next gate", "",
                  "The semantic sanity query works, so this is not evidence that the published "
                  "head cannot make any useful decision. It does show that a working typed interface "
                  "and low latency do not establish Sokoban spatial planning. Neither coordinates "
                  "nor using the pretrained CHOICE head was sufficient under this protocol.", "",
                  "Do not extend training merely from this result: there was no training here. "
                  "A separate, useful next experiment would establish a learnability gate on tiny "
                  "boards with game-specific CHOICE supervision, variable action counts and "
                  "option-order augmentation, then evaluate unseen layouts before scaling episodes. "
                  "Keep any RLCD objective comparison separate from the representation/learnability check.", "",
                  "This small generated suite does not test hard handcrafted Sokoban, the model's "
                  "unknown original training corpus, another prompt family, tuned game training, or "
                  "TypeSafe Jev. Ordering replicates are correlated, and no broad statistical "
                  "superiority/inferiority claim is justified. A later study needs new layouts "
                  "rather than tuning on these reported evaluation boards.", "",
                  "## Artifacts", "",
                  "Raw per-step traces and logs: `runs/laya-choice-inference-20260922/` on the "
                  "local checkout and server (ignored in Git). The adjacent JSON receipt preserves "
                  "the frozen levels, runtime, aggregates, per-episode outcomes and SHA-256 hashes "
                  "of every raw JSON artifact. No paid API was used.", "",
                  "```bash",
                  "uv run python scripts/summarize_choice_inference.py runs/laya-choice-inference-20260922 "
                  "--output docs/LAYA_CHOICE_INFERENCE_RESULTS.md",
                  "```", ""])
    receipt = {
        "summary": summary, "runtime": runtime, "groups": groups,
        "primitive_replayed_episodes": sum(len(r["results"]) for r in reports.values()),
        "levels": list(levels.values()),
        "outcomes": {name: [{k: v for k, v in row.items() if k != "trace"} for row in report["results"]]
                     for name, report in reports.items()},
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.json"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified_episodes": receipt["primitive_replayed_episodes"], "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
