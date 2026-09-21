"""Solver-trajectory evidence for the Sokoban push task."""

from __future__ import annotations

from pathlib import Path

from jevgames.contracts import CalibrationDecision, DecisionTask, EvidenceProvider
from jevgames.registry import register_evidence
from sokoban_laya.core import Board
from sokoban_laya.evidence import compress_expert_trajectories
from sokoban_laya.trajectories import read_jsonl


class SokobanSolverEvidence(EvidenceProvider):
    name = "sokoban_solver"

    def __init__(self, config: dict) -> None:
        if config:
            unknown = ", ".join(sorted(config))
            raise ValueError(f"sokoban_solver evidence has no configurable fields: {unknown}")

    def load(self, dataset_path: str | Path, task: DecisionTask) -> list[CalibrationDecision]:
        if task.name != "sokoban_push":
            raise ValueError("sokoban_solver evidence requires task.type = 'sokoban_push'")
        macro_examples = compress_expert_trajectories(read_jsonl(dataset_path))
        decisions = []
        for example in macro_examples:
            board = Board.from_ascii(example.board)
            decisions.append(CalibrationDecision.one_hot(
                serialized_state=example.board,
                observation=task.observation(board),
                instruction=task.instruction,
                options=task.options(board),
                target_key=example.expert_label,
                evidence_id=example.episode,
                step=example.push_step,
                provenance="solver_solution",
            ))
        return decisions


@register_evidence("sokoban_solver")
def create_sokoban_solver_evidence(config: dict) -> SokobanSolverEvidence:
    return SokobanSolverEvidence(config)
