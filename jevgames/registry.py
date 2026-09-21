"""Explicit plugin registry with lazy built-in discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import (
    DecisionModelAdapter,
    DecisionTask,
    EvidenceCollector,
    EvidenceProvider,
    TrainingStrategy,
)


ModelFactory = Callable[[dict[str, Any]], DecisionModelAdapter]
TaskFactory = Callable[[dict[str, Any]], DecisionTask]
StrategyFactory = Callable[[dict[str, Any]], TrainingStrategy]
EvidenceFactory = Callable[[dict[str, Any]], EvidenceProvider]
CollectorFactory = Callable[[dict[str, Any]], EvidenceCollector]

_MODELS: dict[str, ModelFactory] = {}
_TASKS: dict[str, TaskFactory] = {}
_STRATEGIES: dict[str, StrategyFactory] = {}
_EVIDENCE: dict[str, EvidenceFactory] = {}
_COLLECTORS: dict[str, CollectorFactory] = {}
_BUILTINS_LOADED = False


def register_model(name: str):
    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _MODELS:
            raise ValueError(f"model plugin {name!r} is already registered")
        _MODELS[name] = factory
        return factory
    return decorator


def register_task(name: str):
    def decorator(factory: TaskFactory) -> TaskFactory:
        if name in _TASKS:
            raise ValueError(f"task plugin {name!r} is already registered")
        _TASKS[name] = factory
        return factory
    return decorator


def register_strategy(name: str):
    def decorator(factory: StrategyFactory) -> StrategyFactory:
        if name in _STRATEGIES:
            raise ValueError(f"training strategy {name!r} is already registered")
        _STRATEGIES[name] = factory
        return factory
    return decorator


def register_evidence(name: str):
    def decorator(factory: EvidenceFactory) -> EvidenceFactory:
        if name in _EVIDENCE:
            raise ValueError(f"evidence plugin {name!r} is already registered")
        _EVIDENCE[name] = factory
        return factory
    return decorator


def register_collector(name: str):
    def decorator(factory: CollectorFactory) -> CollectorFactory:
        if name in _COLLECTORS:
            raise ValueError(f"collector plugin {name!r} is already registered")
        _COLLECTORS[name] = factory
        return factory
    return decorator


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .collectors import episodic as _episodic  # noqa: F401
    from .evidence import sokoban_solver as _sokoban_solver  # noqa: F401
    from .models import laya as _laya  # noqa: F401
    from .models import qwen as _qwen  # noqa: F401
    from .strategies import rlcd as _rlcd  # noqa: F401
    from .tasks import sokoban as _sokoban  # noqa: F401
    _BUILTINS_LOADED = True


def create_model(name: str, config: dict[str, Any]) -> DecisionModelAdapter:
    _load_builtins()
    if name not in _MODELS:
        raise KeyError(f"unknown model plugin {name!r}; available: {sorted(_MODELS)}")
    return _MODELS[name](config)


def create_task(name: str, config: dict[str, Any]) -> DecisionTask:
    _load_builtins()
    if name not in _TASKS:
        raise KeyError(f"unknown task plugin {name!r}; available: {sorted(_TASKS)}")
    return _TASKS[name](config)


def create_strategy(name: str, config: dict[str, Any]) -> TrainingStrategy:
    _load_builtins()
    if name not in _STRATEGIES:
        raise KeyError(f"unknown training strategy {name!r}; available: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](config)


def create_evidence(name: str, config: dict[str, Any]) -> EvidenceProvider:
    _load_builtins()
    if name not in _EVIDENCE:
        raise KeyError(f"unknown evidence plugin {name!r}; available: {sorted(_EVIDENCE)}")
    return _EVIDENCE[name](config)


def create_collector(name: str, config: dict[str, Any]) -> EvidenceCollector:
    _load_builtins()
    if name not in _COLLECTORS:
        raise KeyError(f"unknown collector plugin {name!r}; available: {sorted(_COLLECTORS)}")
    return _COLLECTORS[name](config)


def available_plugins() -> dict[str, list[str]]:
    _load_builtins()
    return {
        "collectors": sorted(_COLLECTORS),
        "evidence": sorted(_EVIDENCE),
        "models": sorted(_MODELS),
        "strategies": sorted(_STRATEGIES),
        "tasks": sorted(_TASKS),
    }
