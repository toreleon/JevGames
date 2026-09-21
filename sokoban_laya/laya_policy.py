"""Laya inference adapter for the solver-supervised Sokoban action policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .core import ACTION_DESCRIPTIONS, Action, Board
from .device import resolve_device

BOARD_LEGEND = "# wall; . goal; $ box; * box on goal; @ player; + player on goal"
QUESTION_INSTRUCTION = (
    "Choose the legal Sokoban action that best advances toward putting every box on a goal. "
    "Do not create a deadlock by pushing a box into a non-goal corner."
)


class PredictingAgent(Protocol):
    def predict(self, state: dict[str, str], questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: Action
    confidence: float
    probability: float
    probabilities: Mapping[str, float]


def question_for(legal_actions: Sequence[Action]) -> dict[str, Any]:
    """Create a compact, state-specific typed-choice question."""

    if len(legal_actions) < 2:
        raise ValueError("Laya requires at least two options; handle forced moves in the controller")
    return {
        "type": "choice",
        "instructions": QUESTION_INSTRUCTION,
        "criteria": {action.value: ACTION_DESCRIPTIONS[action] for action in legal_actions},
    }


class LayaPolicy:
    """A closed-loop next-action policy; the caller remains the game controller."""

    def __init__(self, agent: PredictingAgent) -> None:
        self._agent = agent

    @classmethod
    def from_hub(
        cls,
        model_id_or_path: str = "convaiinnovations/laya",
        *,
        subfolder: str | None = "typed-decisions",
        device: str | None = "auto",
    ) -> "LayaPolicy":
        """Load Laya lazily, so the non-ML tools have no heavyweight import."""

        try:
            import laya
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the optional model dependencies: pip install -e '.[laya]'") from exc
        return cls(laya.load(model_id_or_path, subfolder=subfolder, device=str(resolve_device(device))))

    def choose(self, board: Board) -> PolicyDecision:
        legal = board.legal_actions()
        if not legal:
            raise RuntimeError("Sokoban position has no legal non-no-op action")
        # DecisionModel derives uncertainty features with topk(2).  A forced
        # move is not a model decision at all, so executing it directly both
        # avoids that shape constraint and removes pointless training rows.
        if len(legal) == 1:
            return PolicyDecision(legal[0], confidence=1.0, probability=1.0, probabilities={legal[0].value: 1.0})
        question = question_for(legal)
        result = self._agent.predict(
            {"board": board.render(), "legend": BOARD_LEGEND},
            {"next_action": question},
        )
        answer = result["answers"]["next_action"]
        action = Action(answer["choice"])
        if action not in legal:
            raise RuntimeError("Laya returned an action not included in the legal-action question")
        probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
        return PolicyDecision(
            action=action,
            confidence=float(answer["confidence"]),
            probability=probabilities[action.value],
            probabilities=probabilities,
        )
