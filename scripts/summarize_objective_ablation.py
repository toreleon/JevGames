"""Write a compact, auditable result note from the frozen paired study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    summary = json.loads((root / "summary.json").read_text())
    config = summary["config"]
    reports = {(method, seed): json.loads((root / f"{method}-{seed}" / "report.json").read_text())
               for method in config["study"]["objectives"] for seed in config["study"]["seeds"]}
    prior = json.loads((root / "prior_baselines.json").read_text())
    random = json.loads((root / "random_baselines.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    test_scores = summary["splits"]["test"]["proper_score"]
    differences = test_scores["paired_rlcd_minus_direct"]
    winner = ("direct proper-score optimization" if all(v < 0 for v in differences)
              else "RLCD" if all(v > 0 for v in differences) else "neither method consistently")
    lines = [
        "# Paired objective experiment results — 2026-09-21", "",
        f"At the registered budget, {winner} wins the paired test proper-score comparison. "
        "The tables below compare both trained methods with untrained, random and "
        "constant-prior controls; a forecasting win alone does not establish useful planning.", "",
        "Fixed protocol: [OBJECTIVE_ABLATION.md](OBJECTIVE_ABLATION.md). "
        "These results concern this implementation, architecture, small dataset and update budget.", "",
        f"Three paired seeds {config['study']['seeds']}; 512 training rows from 64 fresh layouts, "
        "128 optimizer updates per arm. Identical initialization hashes and update counts were verified. "
        "No checkpoint selection, online replay or temperature scaling was applied.", "",
        "| Split | Levels | Evidence | Positive labels |",
        "|---|---:|---:|---:|",
    ]
    for split, entry in manifest["splits"].items():
        lines.append(f"| {split} | {entry['levels']} | {entry['evidence']} | {entry['labels'].get('1', 0)} |")
    for split in ("validation", "test"):
        lines.extend(["", f"## {split.title()}", "",
                      "Mean ± sample SD across three training seeds; higher proper score and solve rate "
                      "are better, lower NLL/Brier are better. Rows from one layout are correlated.", "",
                      "| Predictor | Proper score | NLL | Brier | Solved levels per seed |",
                      "|---|---:|---:|---:|---|"])
        for method in config["study"]["objectives"]:
            metrics = summary["splits"][split]
            formatted = [f"{metrics[name][method]['mean']:.4f} ± {metrics[name][method]['seed_sd']:.4f}"
                         for name in ("proper_score", "nll", "brier")]
            solved = [reports[(method, s)]["after"][split]["environment"]["solved"]
                      for s in config["study"]["seeds"]]
            lines.append(f"| {method} | " + " | ".join(formatted) + f" | {solved} |")
        seeds = config["study"]["seeds"]
        baseline_rows = [reports[("rlcd", s)]["baseline"][split] for s in seeds]
        baseline_values = []
        for metric in ("mean_proper_score", "negative_log_likelihood", "brier_score"):
            values = [row["calibration"][metric] for row in baseline_rows]
            baseline_values.append(f"{fmean(values):.4f} ± {stdev(values):.4f}")
        lines.append("| Untrained same heads | " + " | ".join(baseline_values)
                     + f" | {[r['environment']['solved'] for r in baseline_rows]} |")
        p = prior[split]
        lines.append(f"| Constant train prior | {p['proper_score']:.4f} | {p['nll']:.4f} | {p['brier']:.4f} | — |")
        lines.append(f"| Uniform random actions | — | — | — | {[r['solved'] for r in random[split]]} |")
        margins = summary["splits"][split]["proper_score"]["paired_rlcd_minus_direct"]
        lines.extend(["", f"Paired proper-score differences (RLCD − direct): {margins}.", ""])
    test = summary["splits"]["test"]
    margins = test["proper_score"]["paired_rlcd_minus_direct"]
    brier_change = fmean(test["brier"]["paired_rlcd_minus_direct"])
    success = all(d > 0 for d in margins) and fmean(margins) >= .02 and brier_change <= 0
    solved_gain = fmean(test["solve_rate"]["paired_rlcd_minus_direct"]) * config["dataset"]["test_levels"]
    lines.extend(["## Registered criterion", "",
                  f"RLCD promise criterion met: **{success}**. "
                  f"Mean proper-score difference {fmean(margins):.4f}, "
                  f"mean Brier difference {brier_change:.4f}.", "",
                  f"Mean solved-level difference (RLCD − direct): {solved_gain:.2f}/32. "
                  f"The separate gameplay threshold (+2 levels) was met: **{solved_gain >= 2}**.", "",
                  "## Execution receipts", "",
                  "| Method | Seed | Updates | Training seconds | Peak allocated GiB | Reload max probability error |",
                  "|---|---:|---:|---:|---:|---:|"])
    for (method, seed), report in reports.items():
        stats = report["training_stats"]
        peak = max(s.get("peak_cuda_allocated_gib", 0) for s in stats)
        lines.append(f"| {method} | {seed} | {stats[-1]['optimizer_steps']} | "
                     f"{stats[-1]['seconds']:.1f} | {peak:.3f} | {report['reload_max_probability_error']:.3g} |")
    lines.extend(["", "Each saved checkpoint was reloaded and used for one held-out episode. "
                  "Per-seed reports preserve per-layout forecasts and gameplay outcomes. "
                  "A constant-prior win alone is not evidence of spatial reasoning. "
                  "The experiment uses small one/two-box levels; conclusions do not extend to hard Sokoban "
                  "or independently tuned versions of either objective.", ""])
    changed = [(method, seed) for (method, seed), report in reports.items()
               if report["after"]["test"]["environment"]["records"][0]
               != report["reload_episode"]["records"][0]]
    lines.extend([
        "## Precision and attribution limits", "",
        "Training and post-training evaluation use the Accelerate BF16 forward wrapper. "
        "Native reload uses BF16 backbone storage with FP32 heads, and the checkpoint "
        "serializes heads as FP16. A small probability error can change action ordering "
        "when candidate scores nearly tie. Reload is therefore a functional/approximate "
        "numerical check, not proof of bit-identical trajectories.", "",
        f"Arms whose checked episode trace differed after reload: {changed}.", "",
        "The large raw calibration gap should not be confused with evidence that either "
        "model has learned useful planning. Treat small solved-level differences as "
        "exploratory, given the small sample and precision sensitivity.", "",
        "A separate [constant-input diagnostic](ADVANTAGE_NORMALIZATION_DIAGNOSTIC.md) "
        "isolates a calibration failure of the per-example reward standardization. "
        "It was added after the registered study began, without changing these runs.", "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    receipt = {
        "summary": summary,
        "dataset_hashes": {name: entry["sha256"] for name, entry in manifest["splits"].items()},
        "constant_prior": prior,
        "initialization_hashes": {str(seed): reports[("rlcd", seed)]["initial_fingerprint"]
                                  for seed in config["study"]["seeds"]},
        "runtime": reports[("rlcd", config["study"]["seeds"][0])]["runtime"],
        "bernoulli_control": json.loads((root / "bernoulli_control.json").read_text()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
