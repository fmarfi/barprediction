"""Algorithmic bar sources.

Deliberately simple. They exist to prove the interface is pluggable and to
give a reference scenario to edit from -- not because a random walk predicts
BIST. Anything fitted (ARIMA, gradient boosting, a sequence model) drops in
here as another Predictor subclass with no change to the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Param, Predictor


def _returns(history: pd.DataFrame, lookback: int) -> np.ndarray:
    close = history["Close"].astype("float64")
    r = np.log(close / close.shift(1)).dropna().to_numpy()
    return r[-lookback:] if r.size > lookback else r


class DriftPredictor(Predictor):
    name = "Auto - drift"
    description = (
        "Continues the average log-return of the lookback window, with bar "
        "ranges scaled to recent realised volatility. Deterministic."
    )
    editable = True
    params = (
        Param("lookback", "Lookback bars", 60, "int", min=10, max=500, step=10),
        Param(
            "damping",
            "Drift damping",
            1.0,
            "float",
            min=0.0,
            max=2.0,
            step=0.1,
            help="0 flattens the drift entirely, 1 continues it as measured.",
        ),
    )

    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        r = _returns(history, int(self.config["lookback"]))
        if r.size == 0:
            r = np.array([0.0])

        mu = float(np.mean(r)) * float(self.config["damping"])
        sigma = float(np.std(r)) or 0.005
        close = float(history["Close"].iloc[-1])
        vol = float(history["Volume"].tail(20).mean() or 0.0)

        rows = []
        prev = close
        for _ in index:
            nxt = prev * np.exp(mu)
            # Typical bar reaches ~1 sigma beyond its body on each side.
            hi = max(prev, nxt) * (1.0 + sigma)
            lo = min(prev, nxt) * (1.0 - sigma)
            rows.append(
                {"Open": prev, "High": hi, "Low": lo, "Close": nxt, "Volume": vol}
            )
            prev = nxt
        return self._frame(index, rows)


class RandomWalkPredictor(Predictor):
    name = "Auto - random walk"
    description = (
        "Draws each bar's return from a normal fitted to the lookback window. "
        "Change the seed to resample a different path."
    )
    editable = True
    params = (
        Param("lookback", "Lookback bars", 60, "int", min=10, max=500, step=10),
        Param("seed", "Random seed", 0, "int", min=0, max=9999, step=1),
        Param(
            "vol_mult",
            "Volatility multiplier",
            1.0,
            "float",
            min=0.1,
            max=4.0,
            step=0.1,
        ),
    )

    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        r = _returns(history, int(self.config["lookback"]))
        mu = float(np.mean(r)) if r.size else 0.0
        sigma = (float(np.std(r)) or 0.005) * float(self.config["vol_mult"])

        rng = np.random.default_rng(int(self.config["seed"]))
        close = float(history["Close"].iloc[-1])
        vol = float(history["Volume"].tail(20).mean() or 0.0)

        rows = []
        prev = close
        for _ in index:
            nxt = prev * np.exp(rng.normal(mu, sigma))
            # Intrabar extension, independent of the close-to-close move.
            hi = max(prev, nxt) * (1.0 + abs(rng.normal(0, sigma)))
            lo = min(prev, nxt) * (1.0 - abs(rng.normal(0, sigma)))
            rows.append(
                {"Open": prev, "High": hi, "Low": lo, "Close": nxt, "Volume": vol}
            )
            prev = nxt
        return self._frame(index, rows)


class RepeatPredictor(Predictor):
    name = "Auto - repeat last N"
    description = (
        "Replays the shape of the most recent bars, rebased to the last "
        "close. Useful for testing a repeating pattern or seasonality idea."
    )
    editable = True
    params = (
        Param(
            "offset",
            "Replay starting this many bars back",
            0,
            "int",
            min=0,
            max=250,
            step=1,
            help="0 replays the bars immediately before now.",
        ),
    )

    def propose(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(index)
        off = int(self.config["offset"])
        end = len(history) - off
        start = max(0, end - n)
        window = history.iloc[start:end]

        if window.empty:
            window = history.tail(1)

        # Rebase the window's returns onto the current close.
        base = float(window["Close"].iloc[0])
        scale = float(history["Close"].iloc[-1]) / (base or 1.0)

        rows = []
        for i in range(n):
            src = window.iloc[i % len(window)]
            rows.append(
                {
                    "Open": float(src["Open"]) * scale,
                    "High": float(src["High"]) * scale,
                    "Low": float(src["Low"]) * scale,
                    "Close": float(src["Close"]) * scale,
                    "Volume": float(src["Volume"]),
                }
            )
        return self._frame(index, rows)
