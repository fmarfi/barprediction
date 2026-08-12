"""Hand-drawn bars -- the default source.

These seed the editable table in the dashboard. The point is to give you a
sane starting shape that you then drag into whatever scenario you want to
test, not to be a forecast.
"""

from __future__ import annotations

import pandas as pd

from .base import Param, Predictor


class FlatPredictor(Predictor):
    name = "Manual - flat"
    description = (
        "Every future bar repeats the last close. The neutral canvas: edit "
        "the table to draw your own scenario."
    )
    editable = True
    params = (
        Param(
            "wick_pct",
            "Wick size (% of last close)",
            0.0,
            "float",
            min=0.0,
            max=10.0,
            step=0.1,
            help="Stretches high and low above and below the flat close, so "
            "each bar is a visible doji you can see and click. Open and "
            "close stay equal, so this adds no direction.",
        ),
    )

    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        close = float(history["Close"].iloc[-1])
        vol = float(history["Volume"].tail(20).mean() or 0.0)
        # Open == Close, so the body has no height; this only sets the wicks.
        pad = close * float(self.config["wick_pct"]) / 100.0

        rows = []
        for _ in index:
            rows.append(
                {
                    "Open": close,
                    "High": close + pad,
                    "Low": close - pad,
                    "Close": close,
                    "Volume": vol,
                }
            )
        return self._frame(index, rows)


class RampPredictor(Predictor):
    name = "Manual - ramp"
    description = (
        "A straight move of N% spread evenly over the horizon. Quick way to "
        "ask 'what do my indicators do if it grinds up 5% from here?'"
    )
    editable = True
    params = (
        Param(
            "total_pct",
            "Total move over horizon (%)",
            5.0,
            "float",
            min=-50.0,
            max=50.0,
            step=0.5,
        ),
        Param(
            "wick_pct",
            "Wick size (% of bar)",
            0.4,
            "float",
            min=0.0,
            max=5.0,
            step=0.1,
        ),
    )

    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        close = float(history["Close"].iloc[-1])
        vol = float(history["Volume"].tail(20).mean() or 0.0)
        n = len(index)
        total = float(self.config["total_pct"]) / 100.0
        wick = float(self.config["wick_pct"]) / 100.0

        # Compound per-bar rate so the final bar lands exactly on the target.
        rate = (1.0 + total) ** (1.0 / n) - 1.0 if n else 0.0

        rows = []
        prev = close
        for _ in index:
            nxt = prev * (1.0 + rate)
            hi = max(prev, nxt) * (1.0 + wick)
            lo = min(prev, nxt) * (1.0 - wick)
            rows.append(
                {"Open": prev, "High": hi, "Low": lo, "Close": nxt, "Volume": vol}
            )
            prev = nxt
        return self._frame(index, rows)
