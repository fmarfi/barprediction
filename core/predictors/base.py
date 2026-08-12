"""The bar-source interface.

A Predictor proposes the OHLCV for the next `len(index)` bars. That is the
whole contract. The dashboard never cares whether the numbers came from you
typing them, from a random walk, or from a trained model -- it just charts
them and runs indicators over history + proposal.

To add a new source: subclass Predictor, implement `propose`, and register it
in core/predictors/__init__.py.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class Param:
    """One tunable knob, rendered automatically as a dashboard control."""

    key: str
    label: str
    default: Any
    kind: str = "float"  # float | int | bool | choice
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""


class Predictor(abc.ABC):
    """Base class for anything that can propose future bars."""

    name: str = "unnamed"
    description: str = ""
    #: True when the user is expected to hand-edit the result afterwards.
    editable: bool = True
    params: tuple[Param, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        self.config = {p.key: kwargs.get(p.key, p.default) for p in self.params}

    @abc.abstractmethod
    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a frame indexed by `index` with the COLUMNS above."""

    # -- helpers available to subclasses ----------------------------------

    @staticmethod
    def _frame(index: pd.DatetimeIndex, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, index=index, columns=COLUMNS).astype("float64")
        df.index.name = "Date"
        return sanitize(df)


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Force OHLC consistency: high is the max, low is the min.

    Hand-entered bars routinely violate this, and an inverted bar makes the
    stochastic and SAR produce nonsense rather than an obvious error.
    """
    out = df.copy()
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    out = out[COLUMNS].astype("float64")
    # Derive both extremes from the original four values. Computing Low after
    # overwriting High would feed the corrected High back in and lose a low
    # that was typed above the body.
    body = out[["Open", "High", "Low", "Close"]].copy()
    out["High"] = body.max(axis=1)
    out["Low"] = body.min(axis=1)
    out["Volume"] = out["Volume"].clip(lower=0.0)
    return out
