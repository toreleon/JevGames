"""Command-line interface for solving, data generation, and policy rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import Action, Board
from .laya_policy import LayaPolicy
from .solver import solve
from .trajectories import solution_trajectory, write_jsonl


def _load_level(path: str) -> Board:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    return Board.from_xsb(text) if source.suffix.lower() == ".xsb" else Board.from_ascii(text)


def _print_actions(actions: Sequence[Action]) -> None:
    print(" ".join(action.value for action in actions) or "(already solved)")


def command_solve(args: argparse.Namespace) -> int:
    board = _load_level(args.level)
    result = solve(board, max_expansions=args.max_expansions)
    if result is None:
        print("No solution found within the expansion limit.")
        return 2
    print(f"Solved in {len(result.actions)} actions / {result.pushes} pushes ({result.expanded} states expanded).")
    _print_actions(result.actions)
    if args.trajectory:
        examples = solution_trajectory(board, result, Path(args.level).stem)
        write_jsonl(examples, args.trajectory)
        print(f"Wrote {len(examples)} expert decisions to {args.trajectory}")
    return 0


def command_play(args: argparse.Namespace) -> int:
    board = _load_level(args.level)
    policy = LayaPolicy.from_hub(args.model, subfolder=None, device=args.device)
    seen: set[Board] = set()
    print(board.render())
    for step in range(args.max_steps):
        if board.is_solved():
            print(f"Solved in {step} actions.")
            return 0
        use_fallback = board in seen or board.has_static_deadlock()
        decision = None
        if not use_fallback:
            decision = policy.choose(board)
            use_fallback = decision.confidence < args.min_confidence
        if use_fallback:
            result = solve(board, max_expansions=args.max_expansions)
            if result is None or not result.actions:
                print("Policy entered an unsolved position and fallback search failed.")
                return 2
            action = result.actions[0]
            source = "solver fallback"
        else:
            action = decision.action
            source = f"Laya confidence={decision.confidence:.3f}"
        seen.add(board)
        transition = board.apply(action)
        if not transition.moved:
            print(f"Controller rejected invalid action {action.value!r}.")
            return 2
        board = transition.board
        print(f"\n{step + 1:03d}: {action.value} ({source})\n{board.render()}")
    print("Step limit reached without solving the level.")
    return 2


def command_train(args: argparse.Namespace) -> int:
    from .training import FineTuneConfig, fine_tune

    config = FineTuneConfig(epochs=args.epochs, batch_size=args.batch_size)
    destination = fine_tune(args.dataset, args.output, config, device=args.device)
    print(f"Saved fine-tuned Laya model to {destination}")
    return 0


def command_train_grpo(args: argparse.Namespace) -> int:
    from .grpo_training import GRPOConfig, train_environment_grpo

    config = GRPOConfig(
        iterations=args.iterations,
        group_size=args.group_size,
        max_steps=args.max_steps,
        max_levels=args.max_levels,
        batch_size=args.batch_size,
    )
    destination = train_environment_grpo(args.model, args.dataset, args.output, config, device=args.device)
    print(f"Saved environment-GRPO Laya model to {destination}")
    return 0


def command_train_macro_grpo(args: argparse.Namespace) -> int:
    from .macro_grpo_training import MacroGRPOConfig, train_macro_grpo

    config = MacroGRPOConfig(
        expert_warmup_epochs=args.expert_warmup_epochs,
        iterations=args.iterations,
        group_size=args.group_size,
        max_pushes=args.max_pushes,
        max_levels=args.max_levels,
        batch_size=args.batch_size,
        rollout_level_batch_size=args.rollout_level_batch_size,
        rollout_inference_batch_size=args.rollout_inference_batch_size,
    )
    destination = train_macro_grpo(args.model, args.dataset, args.output, config, device=args.device)
    print(f"Saved push-macro GRPO Laya model to {destination}")
    return 0


def command_benchmark_macro(args: argparse.Namespace) -> int:
    import json
    from .benchmark import BenchmarkConfig, evaluate_macro_dataset, save_benchmark

    result = evaluate_macro_dataset(
        args.model,
        args.dataset,
        BenchmarkConfig(args.device, args.max_levels, args.max_pushes, args.batch_size),
        args.manifest,
    )
    print(json.dumps(result, sort_keys=True))
    if args.output:
        save_benchmark(result, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy Laya/Sokoban compatibility CLI for Jev Games")
    commands = parser.add_subparsers(dest="command", required=True)

    solve_parser = commands.add_parser("solve", help="solve a level with A* and optionally export JSONL demonstrations")
    solve_parser.add_argument("level", help="path to an ASCII Sokoban level")
    solve_parser.add_argument("--trajectory", help="write solver decisions to this JSONL path")
    solve_parser.add_argument("--max-expansions", type=int, default=200_000)
    solve_parser.set_defaults(handler=command_solve)

    play_parser = commands.add_parser("play", help="run a fine-tuned Laya policy in closed loop")
    play_parser.add_argument("level", help="path to an ASCII Sokoban level")
    play_parser.add_argument("--model", required=True, help="local fine-tuned model directory or Hugging Face model id")
    play_parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    play_parser.add_argument("--max-steps", type=int, default=200)
    play_parser.add_argument("--max-expansions", type=int, default=200_000)
    play_parser.add_argument("--min-confidence", type=float, default=0.70)
    play_parser.set_defaults(handler=command_play)

    train_parser = commands.add_parser("train", help="legacy notebook-compatible RLCD baseline")
    train_parser.add_argument("dataset")
    train_parser.add_argument("output")
    train_parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.set_defaults(handler=command_train)

    grpo_parser = commands.add_parser("train-grpo", help="legacy primitive-action GRPO ablation")
    grpo_parser.add_argument("model", help="warm-start Laya checkpoint directory or Hugging Face model id")
    grpo_parser.add_argument("dataset", help="solver-labelled JSONL used for initial levels and expert replay")
    grpo_parser.add_argument("output", help="destination checkpoint directory")
    grpo_parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    grpo_parser.add_argument("--iterations", type=int, default=1)
    grpo_parser.add_argument("--group-size", type=int, default=4)
    grpo_parser.add_argument("--max-steps", type=int, default=32)
    grpo_parser.add_argument("--max-levels", type=int, default=4)
    grpo_parser.add_argument("--batch-size", type=int, default=4)
    grpo_parser.set_defaults(handler=command_train_grpo)

    macro_parser = commands.add_parser("train-macro-grpo", help="run GRPO where Laya selects reachable box pushes and BFS handles walking")
    macro_parser.add_argument("model", help="warm-start Laya checkpoint directory or Hugging Face model id")
    macro_parser.add_argument("dataset", help="primitive solver JSONL; it is compressed into push demonstrations")
    macro_parser.add_argument("output", help="destination checkpoint directory")
    macro_parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    macro_parser.add_argument("--iterations", type=int, default=2)
    macro_parser.add_argument("--expert-warmup-epochs", type=int, default=6)
    macro_parser.add_argument("--group-size", type=int, default=4)
    macro_parser.add_argument("--max-pushes", type=int, default=32)
    macro_parser.add_argument("--max-levels", type=int, default=4)
    macro_parser.add_argument("--batch-size", type=int, default=4)
    macro_parser.add_argument("--rollout-level-batch-size", type=int, default=16)
    macro_parser.add_argument("--rollout-inference-batch-size", type=int, default=32)
    macro_parser.set_defaults(handler=command_train_macro_grpo)

    benchmark_parser = commands.add_parser("benchmark-macro", help="run model-only batched evaluation on a held-out JSONL split")
    benchmark_parser.add_argument("model")
    benchmark_parser.add_argument("dataset")
    benchmark_parser.add_argument("--manifest")
    benchmark_parser.add_argument("--output")
    benchmark_parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    benchmark_parser.add_argument("--max-levels", type=int)
    benchmark_parser.add_argument("--max-pushes", type=int, default=32)
    benchmark_parser.add_argument("--batch-size", type=int, default=32)
    benchmark_parser.set_defaults(handler=command_benchmark_macro)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
