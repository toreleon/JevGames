"""Paired, fixed-budget RLCD versus direct proper-score experiment.

Run each arm in a fresh process. This prevents Accelerator mixed-precision
wrappers and optimizer state from leaking between trials.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
import subprocess
import sys
import time
import tomllib

from jevgames.config import BenchmarkConfig
from jevgames.engine import evaluate_calibration
from jevgames.registry import create_model, create_strategy
from jevgames.research.bounded_sokoban import (
    encode_record, generate_dataset, observation, question, state_key, successor,
)
from sokoban_laya.core import Board
from sokoban_laya.macros import legal_push_macros


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def save_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def constant_baseline(rows, p):
    """No spatial reasoning: predict the training prevalence for every query."""
    norm = math.sqrt(p * p + (1 - p)**2)
    return {
        "predicted_p_true": p,
        "accuracy": fmean((p >= .5) == row["label"] for row in rows),
        "brier": fmean(2 * (p - int(row["label"]))**2 for row in rows),
        "nll": fmean(-math.log(p if row["label"] else 1 - p) for row in rows),
        "proper_score": fmean(
            math.log(p if row["label"] else 1 - p) + .5 * (p if row["label"] else 1 - p) / norm
            for row in rows
        ),
    }


def to_device(batch, device):
    import torch
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def probabilities(adapter, items, batch_size):
    import torch
    values = []
    adapter.model.eval()
    with torch.no_grad():
        for offset in range(0, len(items), batch_size):
            batch = to_device(adapter.collate(items[offset:offset + batch_size]), adapter.device)
            values.extend(torch.softmax(adapter.logits(adapter.model, batch), -1)[:, 1].float().cpu().tolist())
    return values


def fingerprint(adapter):
    digest = hashlib.sha256()
    for name, parameter in adapter.model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def evidence_metrics(adapter, payload, batch_size):
    rows = payload["evidence"]
    items = [encode_record(adapter, row) for row in rows]
    metrics = evaluate_calibration(adapter, items, BenchmarkConfig(batch_size=batch_size))
    predicted = probabilities(adapter, items, batch_size)
    grouped = defaultdict(list)
    for row, probability in zip(rows, predicted):
        grouped[row["group"]].append((row, probability))
    ranking = []
    margins = []
    by_layout = defaultdict(list)
    for group in grouped.values():
        if not any(row["label"] for row, _ in group) or all(row["label"] for row, _ in group):
            continue
        best = max(group, key=lambda entry: entry[1])
        ranking.append(float(best[0]["label"]))
        positive = [p for row, p in group if row["label"]]
        negative = [p for row, p in group if not row["label"]]
        margins.append(fmean(positive) - fmean(negative))
        by_layout[best[0]["layout_id"]].append(float(best[0]["label"]))
    metrics["ranking"] = {
        "mixed_label_groups": len(ranking),
        "success": fmean(ranking) if ranking else None,
        "positive_negative_margin": fmean(margins) if margins else None,
        "by_layout": {key: fmean(values) for key, values in by_layout.items()},
    }
    # Row counts are correlated within layouts; retain per-layout metrics for analysis.
    layout_rows = defaultdict(list)
    for row, p in zip(rows, predicted):
        layout_rows[row["layout_id"]].append((row, p))
    metrics["by_layout"] = {}
    for layout, values in layout_rows.items():
        metrics["by_layout"][layout] = {
            "nll": fmean(-math.log(max(p if row["label"] else 1 - p, 1e-12)) for row, p in values),
            "brier": fmean(2 * (p - int(row["label"]))**2 for row, p in values),
        }
    return metrics


def environment_metrics(adapter, levels, horizon, batch_size, random_seed=0):
    """One fixed greedy controller: no search, deadlock filter or novelty bonus.

    Horizon counts down, exactly matching the supervised event. Repeated boards
    are recorded but are not forcibly terminated: H has changed on revisiting.
    """
    rng = random.Random(random_seed)
    live = [{"board": Board.from_ascii(row["board"]), "id": row["layout_id"],
             "seen": set(), "reason": None, "steps": 0, "revisits": 0, "boxes": row["boxes"]}
            for row in levels]
    for step in range(horizon):
        requests = []
        choices = {}
        for index, episode in enumerate(live):
            if episode["reason"] is not None:
                continue
            board = episode["board"]
            key = state_key(board)
            episode["revisits"] += int(key in episode["seen"])
            episode["seen"].add(key)
            macros = legal_push_macros(board)
            if not macros:
                episode["reason"] = "no_action"
                continue
            if adapter is None:
                choices[index] = rng.choice(macros)
            else:
                for macro in macros:
                    requests.append((index, macro, adapter.encode(
                        observation(board.render()), question(macro.description, horizon - step),
                    )))
        if requests:
            ps = probabilities(adapter, [row[2] for row in requests], batch_size)
            scored = defaultdict(list)
            for (index, macro, _), probability in zip(requests, ps):
                scored[index].append((macro, probability))
            for index, candidates in scored.items():
                choices[index] = max(candidates, key=lambda entry: entry[1])[0]
        for index, macro in choices.items():
            episode = live[index]
            episode["board"] = successor(episode["board"], macro)
            episode["steps"] += 1
            if episode["board"].is_solved():
                episode["reason"] = "solved"
            elif episode["board"].has_static_deadlock():
                episode["reason"] = "static_deadlock"
    records = [{"layout_id": row["id"], "boxes": row["boxes"],
                "outcome": row["reason"] or "horizon", "steps": row["steps"],
                "revisits": row["revisits"]} for row in live]
    return {
        "instances": len(records), "solved": sum(row["outcome"] == "solved" for row in records),
        "solve_rate": fmean(row["outcome"] == "solved" for row in records),
        "mean_revisits": fmean(row["revisits"] for row in records),
        "outcomes": dict(Counter(row["outcome"] for row in records)), "records": records,
    }


def evaluate(adapter, payload, config):
    return {
        "calibration": evidence_metrics(adapter, payload, config["evaluation"]["batch_size"]),
        "environment": environment_metrics(
            adapter, payload["levels"], config["dataset"]["horizon"], config["evaluation"]["batch_size"],
        ),
    }


def run_arm(config, objective, seed):
    import torch
    started = time.perf_counter()
    root = Path(config["study"]["output_dir"])
    directory = root / f"{objective}-{seed}"
    directory.mkdir(exist_ok=False)
    data_path = root / "data"
    manifest = json.loads((data_path / "manifest.json").read_text())
    for split, entry in manifest["splits"].items():
        if hashlib.sha256((data_path / f"{split}.json").read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"dataset hash mismatch: {split}")
    payloads = {split: json.loads((data_path / f"{split}.json").read_text())
                for split in ("train", "validation", "test")}
    adapter = create_model("qwen_decision", dict(config["model"], seed=seed))
    initial_fingerprint = fingerprint(adapter)
    import importlib.metadata
    runtime = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "base_model_revision": getattr(adapter.model.backbone.config, "_commit_hash", None),
    }
    # Check token budget before training; silently truncated states invalidate the study.
    configured_length = adapter.max_length
    adapter.max_length = 4096
    max_tokens = max(
        len(ids) for payload in payloads.values() for row in payload["evidence"]
        for ids in encode_record(adapter, row)["candidate_ids"]
    )
    adapter.max_length = configured_length
    if max_tokens > configured_length:
        raise ValueError(f"prompt truncation: {max_tokens} > {configured_length}")
    training_items = [encode_record(adapter, row) for row in payloads["train"]["evidence"]]
    baseline = {split: evaluate(adapter, payloads[split], config) for split in ("validation", "test")}
    save_json(directory / "baseline.json", baseline)
    emit({"phase": "arm_start", "objective": objective, "seed": seed,
          "fingerprint": initial_fingerprint, "max_tokens": max_tokens})
    strategy = create_strategy(objective, config["training"])
    stats = strategy.train(adapter, training_items, directory, seed)
    # Frozen final-epoch checkpoint: no validation selection or temperature fitting.
    after = {split: evaluate(adapter, payloads[split], config) for split in ("validation", "test")}
    train_metrics = evidence_metrics(adapter, payloads["train"], config["evaluation"]["batch_size"])
    checkpoint = adapter.save(adapter.model, directory / "checkpoint", {
        "objective": objective, "seed": seed, "initial_fingerprint": initial_fingerprint,
        "runtime": runtime,
        "config": config, "stats": stats,
    })
    probe_items = [encode_record(adapter, row) for row in payloads["test"]["evidence"][:16]]
    before_reload = probabilities(adapter, probe_items, 16)
    del adapter, strategy
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = create_model("qwen_decision", {"source": str(checkpoint), "device": config["model"]["device"]})
    after_reload = probabilities(reloaded, probe_items, 16)
    reload_error = max(abs(a - b) for a, b in zip(before_reload, after_reload))
    if reload_error > 0.01:
        raise RuntimeError(f"checkpoint reload probability error {reload_error}")
    replay = environment_metrics(reloaded, payloads["test"]["levels"][:1], config["dataset"]["horizon"], 16)
    report = {
        "objective": objective, "seed": seed, "initial_fingerprint": initial_fingerprint,
        "runtime": runtime,
        "final_fingerprint": fingerprint(reloaded), "max_tokens": max_tokens,
        "training_stats": stats, "baseline": baseline, "after": after,
        "train_calibration": train_metrics, "reload_max_probability_error": reload_error,
        "reload_episode": replay, "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
    }
    save_json(directory / "report.json", report)
    emit({"phase": "arm_complete", "objective": objective, "seed": seed,
          "seconds": report["elapsed_seconds"], "test_brier": after["test"]["calibration"]["brier_score"],
          "test_nll": after["test"]["calibration"]["negative_log_likelihood"],
          "test_solved": after["test"]["environment"]["solved"]})


def aggregate(config):
    root = Path(config["study"]["output_dir"])
    seeds = config["study"]["seeds"]
    metrics = {"proper_score": ("calibration", "mean_proper_score"),
               "nll": ("calibration", "negative_log_likelihood"),
               "brier": ("calibration", "brier_score"),
               "balanced_accuracy": ("calibration", "balanced_accuracy"),
               "solve_rate": ("environment", "solve_rate")}
    reports = {(objective, seed): json.loads((root / f"{objective}-{seed}" / "report.json").read_text())
               for objective in config["study"]["objectives"] for seed in seeds}
    for seed in seeds:
        if reports[("rlcd", seed)]["initial_fingerprint"] != reports[("direct_proper_score", seed)]["initial_fingerprint"]:
            raise RuntimeError("paired arms do not share initialization")
        left = reports[("rlcd", seed)]["training_stats"][-1]["optimizer_steps"]
        right = reports[("direct_proper_score", seed)]["training_stats"][-1]["optimizer_steps"]
        if left != right:
            raise RuntimeError("unequal update budgets")
    summary = {"config": config, "paired_initialization_and_update_budget_verified": True, "splits": {}}
    for split in ("validation", "test"):
        summary["splits"][split] = {}
        for name, path in metrics.items():
            values = {objective: [reports[(objective, seed)]["after"][split][path[0]][path[1]] for seed in seeds]
                      for objective in config["study"]["objectives"]}
            paired = [a - b for a, b in zip(values["rlcd"], values["direct_proper_score"])]
            summary["splits"][split][name] = {
                **{objective: {"values": vals, "mean": fmean(vals), "seed_sd": stdev(vals)}
                   for objective, vals in values.items()},
                "paired_rlcd_minus_direct": paired,
            }
    save_json(root / "summary.json", summary)
    emit({"phase": "study_complete", "summary": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "arm", "summarize"))
    parser.add_argument("config")
    parser.add_argument("--objective", choices=("rlcd", "direct_proper_score"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    root = Path(config["study"]["output_dir"])
    if args.command == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "protocol.json", config)
        manifest = generate_dataset(config["dataset"], root / "data")
        random_baselines = {}
        prior_baselines = {}
        train = json.loads((root / "data" / "train.json").read_text())
        prior = fmean(row["label"] for row in train["evidence"])
        for split in ("validation", "test"):
            payload = json.loads((root / "data" / f"{split}.json").read_text())
            random_baselines[split] = [environment_metrics(
                None, payload["levels"], config["dataset"]["horizon"], 1, seed,
            ) for seed in config["study"]["seeds"]]
            prior_baselines[split] = constant_baseline(payload["evidence"], prior)
        save_json(root / "random_baselines.json", random_baselines)
        save_json(root / "prior_baselines.json", prior_baselines)
        emit({"phase": "prepared", "manifest": manifest})
    elif args.command == "arm":
        if args.objective is None or args.seed is None:
            parser.error("arm requires --objective and --seed")
        run_arm(config, args.objective, args.seed)
    elif args.command == "run":
        if json.loads((root / "protocol.json").read_text()) != config:
            raise ValueError("registered protocol changed; use a new output directory")
        for index, seed in enumerate(config["study"]["seeds"]):
            # Alternate execution order to reduce systematic machine-load bias.
            objectives = config["study"]["objectives"][::1 if index % 2 == 0 else -1]
            for objective in objectives:
                with (root / f"{objective}-{seed}.log").open("x") as logfile:
                    emit({"phase": "dispatch", "objective": objective, "seed": seed})
                    subprocess.run([
                        sys.executable, "-m", "jevgames.research.objective_ablation", "arm", args.config,
                        "--objective", objective, "--seed", str(seed),
                    ], stdout=logfile, stderr=subprocess.STDOUT, check=True)
        aggregate(config)
    else:
        aggregate(config)


if __name__ == "__main__":
    main()
