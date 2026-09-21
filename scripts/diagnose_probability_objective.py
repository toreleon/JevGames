"""A constant-input Bernoulli control for the probability-training objective.

This post-hoc diagnostic does not alter the registered Sokoban study. All
examples have the same input, with exactly 30% true outcomes. The best calibrated
report is therefore p(true)=0.3; spatial representation cannot be a confound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jevgames.scoring import composite_proper_score
from jevgames.strategies.rlcd import DirectProperScoreStrategy, RLCDStrategy


class CenteredScoreFunctionControl(RLCDStrategy):
    """Leave-one-out reward baseline, without label-dependent std normalization."""

    name = "centered_pg_control"

    def objective(self, logits, targets, mask, ordinal, sigma, generator):
        noise = torch.randn(
            (self.config.group_size,) + tuple(logits.shape), generator=generator,
        ) * sigma
        noise -= noise.mean(-1, keepdim=True)
        sampled = logits.detach().unsqueeze(0) + noise
        with torch.no_grad():
            reward = composite_proper_score(torch.softmax(sampled, -1), targets, mask, ordinal)
            # Equivalent to subtracting the other samples' mean reward.
            advantage = (reward - reward.mean(0, keepdim=True)) * self.config.group_size / (self.config.group_size - 1)
        logp = -((sampled - logits.unsqueeze(0))**2).sum(-1) / (2 * sigma**2)
        return -(advantage * logp).mean(), reward, advantage.abs().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    targets = torch.nn.functional.one_hot(torch.tensor([0] * 70 + [1] * 30), 2).float()
    mask = torch.ones_like(targets, dtype=torch.bool)
    ordinal = torch.zeros(100, dtype=torch.bool)
    rows = []
    for seed in (17, 29, 43):
        for cls in (RLCDStrategy, DirectProperScoreStrategy, CenteredScoreFunctionControl):
            logits = torch.nn.Parameter(torch.zeros(2))
            optimizer = torch.optim.AdamW([logits], lr=.03, weight_decay=0)
            generator = torch.Generator().manual_seed(seed)
            method = cls({"group_size": 16})
            for step in range(128):
                sigma = .6 + (.15 - .6) * step / 127
                loss, _, _ = method.objective(logits.expand(100, -1), targets, mask, ordinal, sigma, generator)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            p = torch.softmax(logits.detach(), -1)
            row = {
                "seed": seed, "method": method.name, "true_probability": .3,
                "predicted_probability": float(p[1]),
                "proper_score": float(composite_proper_score(p.expand(100, -1), targets, mask, ordinal).mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "design": "post-hoc constant input; 70 false + 30 true; 128 steps; AdamW lr=.03; no decay",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
