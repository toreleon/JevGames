"""Explicit plugin registry with lazy built-in discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import DecisionModelAdapter, DecisionTask


ModelFactory = Callable[[dict[str, Any]], DecisionModelAdapter]
TaskFactory = Callable[[dict[str, Any]], DecisionTask]

_MODELS: dict[str, ModelFactory] = {}
_TASKS: dict[str, TaskFactory] = {}
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


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .models import laya as _laya  # noqa: F401
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


def available_plugins() -> dict[str, list[str]]:
    _load_builtins()
    return {"models": sorted(_MODELS), "tasks": sorted(_TASKS)}
