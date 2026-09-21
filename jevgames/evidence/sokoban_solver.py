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
                question=task.question(board),
                target_key=example.expert_label,
                evidence_id=example.episode,
                step=example.push_step,
                provenance="solver_solution",
            ))
        return decisions


class SokobanCounterfactualSolverEvidence(EvidenceProvider):
    """Align representative evidence with the per-action NOUL control surface.

    An expert transition proves that its selected action preserves at least one
    solution. A sound static-deadlock detector proves immediate losing actions.
    Other alternatives remain unknown rather than being mislabeled as failures.
    """

    name = "sokoban_counterfactual_solver"

    def __init__(self, config: dict) -> None:
        if config:
            unknown = ", ".join(sorted(config))
            raise ValueError(
                f"sokoban_counterfactual_solver evidence has no configurable fields: {unknown}"
            )

    def load(self, dataset_path: str | Path, task: DecisionTask) -> list[CalibrationDecision]:
        if task.name != "sokoban_push":
            raise ValueError(
                "sokoban_counterfactual_solver evidence requires task.type = 'sokoban_push'"
            )
        macro_examples = compress_expert_trajectories(read_jsonl(dataset_path))
        decisions = []
        for example in macro_examples:
            board = Board.from_ascii(example.board)
            queries = {query.action_key: query.question for query in task.outcome_queries(board)}
            if example.expert_label not in queries:
                raise ValueError(f"expert action {example.expert_label!r} is not executable")
            decisions.append(CalibrationDecision.one_hot(
                serialized_state=example.board,
                observation=task.observation(board),
                question=queries[example.expert_label],
                target_key="true",
                evidence_id=example.episode,
                step=example.push_step,
                provenance="solver_successor_positive",
            ))
            for action_key, question in queries.items():
                if action_key == example.expert_label:
                    continue
                if task.is_known_terminal_failure(board, action_key):
                    decisions.append(CalibrationDecision.one_hot(
                        serialized_state=example.board,
                        observation=task.observation(board),
                        question=question,
                        target_key="false",
                        evidence_id=example.episode,
                        step=example.push_step,
                        provenance="proved_terminal_failure",
                    ))
        return decisions


@register_evidence("sokoban_solver")
def create_sokoban_solver_evidence(config: dict) -> SokobanSolverEvidence:
    return SokobanSolverEvidence(config)


@register_evidence("sokoban_counterfactual_solver")
def create_sokoban_counterfactual_solver_evidence(
    config: dict,
) -> SokobanCounterfactualSolverEvidence:
    return SokobanCounterfactualSolverEvidence(config)
