"""Predictor registry.

Add a new bar source by importing its class and appending it to _SOURCES.
The dashboard picks it up automatically, including its parameter controls.
"""

from __future__ import annotations

from .auto import DriftPredictor, RandomWalkPredictor, RepeatPredictor
from .base import COLUMNS, Param, Predictor, sanitize
from .manual import FlatPredictor, RampPredictor

_SOURCES: tuple[type[Predictor], ...] = (
    FlatPredictor,
    RampPredictor,
    DriftPredictor,
    RandomWalkPredictor,
    RepeatPredictor,
)

REGISTRY: dict[str, type[Predictor]] = {cls.name: cls for cls in _SOURCES}


def names() -> list[str]:
    return list(REGISTRY)


def build(name: str, **kwargs) -> Predictor:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown predictor {name!r}; have {names()}") from None


__all__ = [
    "COLUMNS",
    "REGISTRY",
    "Param",
    "Predictor",
    "build",
    "names",
    "sanitize",
]
