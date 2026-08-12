"""Swing detection, trend lines and Fibonacci levels.

The chain is: find the pivots, take the most recent impulse leg between two
of them, then hang trend lines and Fibonacci levels off that leg. Everything
is derived from one detected swing so the pieces agree with each other --
a Fibonacci grid anchored to a different leg than the trend line it sits
next to is worse than none at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Retracements sit between the end of the leg (0) and its start (1).
RETRACEMENTS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
# Projections continue past the end of the leg, in the direction it ran.
PROJECTIONS = (1.272, 1.414, 1.618, 2.0, 2.618)

GOLDEN = (0.382, 0.5, 0.618)


@dataclass(frozen=True)
class Swing:
    ts: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


@dataclass(frozen=True)
class Impulse:
    """A directional leg between two pivots."""

    start_ts: pd.Timestamp
    start_price: float
    end_ts: pd.Timestamp
    end_price: float

    @property
    def up(self) -> bool:
        return self.end_price >= self.start_price

    @property
    def direction(self) -> str:
        return "up" if self.up else "down"

    @property
    def size(self) -> float:
        return abs(self.end_price - self.start_price)

    @property
    def pct(self) -> float:
        base = abs(self.start_price) or 1.0
        return (self.end_price - self.start_price) / base * 100.0


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Swing]:
    """Fractal pivots: a high with `left` lower highs before and `right` after.

    `right` bars of confirmation means the last few bars can never be
    pivots yet -- that is honest rather than a limitation, since a high is
    only a high once price has failed to beat it.
    """
    if df.empty or left < 1 or right < 1 or len(df) < left + right + 1:
        return []

    high = df["High"].to_numpy(dtype="float64")
    low = df["Low"].to_numpy(dtype="float64")
    idx = df.index

    out: list[Swing] = []
    for i in range(left, len(df) - right):
        window_h = high[i - left : i + right + 1]
        window_l = low[i - left : i + right + 1]
        # Strict against the left side only, so a flat top still registers
        # once rather than at every bar of the plateau.
        if high[i] == window_h.max() and high[i] > high[i - left : i].max():
            out.append(Swing(idx[i], float(high[i]), "high"))
        elif low[i] == window_l.min() and low[i] < low[i - left : i].min():
            out.append(Swing(idx[i], float(low[i]), "low"))
    return out


def last_impulse(
    df: pd.DataFrame, left: int = 3, right: int = 3
) -> Impulse | None:
    """The leg price is currently working on.

    A pivot needs `right` bars of confirmation, so the far end of the live
    leg is never a confirmed pivot -- which is precisely the leg worth
    measuring. The endpoint is therefore the running extreme since the last
    confirmed pivot, and only if nothing has moved since do we fall back to
    the last two confirmed pivots.
    """
    pts = swing_points(df, left, right)
    if not pts:
        return None

    last = pts[-1]
    tail = df.loc[df.index > last.ts]
    if not tail.empty:
        if last.kind == "low":
            price = float(tail["High"].max())
            if price > last.price:
                return Impulse(
                    last.ts, last.price, tail["High"].idxmax(), price
                )
        else:
            price = float(tail["Low"].min())
            if price < last.price:
                return Impulse(
                    last.ts, last.price, tail["Low"].idxmin(), price
                )

    # Nothing beyond the last pivot: use the last two confirmed ones.
    start = next((p for p in reversed(pts[:-1]) if p.kind != last.kind), None)
    if start is None or start.ts >= last.ts:
        return None
    return Impulse(start.ts, start.price, last.ts, last.price)


def manual_impulse(df: pd.DataFrame, start_i: int, end_i: int) -> Impulse | None:
    """A leg between two bar positions, for when the auto pick is wrong."""
    if df.empty:
        return None
    n = len(df)
    a, b = sorted((int(start_i) % n, int(end_i) % n))
    if a == b:
        return None
    # Anchor on the extremes of the chosen bars, in whichever order makes
    # the leg run the way price actually moved between them.
    lo_first = float(df["Close"].iloc[a]) <= float(df["Close"].iloc[b])
    if lo_first:
        return Impulse(
            df.index[a], float(df["Low"].iloc[a]),
            df.index[b], float(df["High"].iloc[b]),
        )
    return Impulse(
        df.index[a], float(df["High"].iloc[a]),
        df.index[b], float(df["Low"].iloc[b]),
    )


def fib_levels(
    imp: Impulse,
    retracements: tuple[float, ...] = RETRACEMENTS,
    projections: tuple[float, ...] = (),
) -> dict[str, float]:
    """Price for each ratio, keyed by a printable label.

    Retracement r runs from the end of the leg (0) back to its start (1).
    Projection r continues past the end, so 1.618 on a rally that ran
    100 -> 200 sits at 261.8.
    """
    a, b = imp.start_price, imp.end_price
    span = b - a

    out: dict[str, float] = {}
    for r in retracements:
        out[f"{r:.3f}".rstrip("0").rstrip(".")] = b - span * r
    for r in projections:
        out[f"{r:.3f}".rstrip('0').rstrip('.')] = b + span * (r - 1.0)
    return out


def trend_line(
    df: pd.DataFrame, left: int = 3, right: int = 3, kind: str = "auto"
) -> tuple[Swing, Swing] | None:
    """Two same-kind pivots defining a line to extend forward.

    Lows give support in an uptrend, highs give resistance in a downtrend;
    "auto" picks whichever matches the last impulse.
    """
    pts = swing_points(df, left, right)
    if len(pts) < 2:
        return None

    if kind == "auto":
        imp = last_impulse(df, left, right)
        kind = "low" if (imp is None or imp.up) else "high"

    same = [p for p in pts if p.kind == kind]
    if len(same) < 2:
        return None
    return same[-2], same[-1]


def project(
    anchors: tuple[Swing, Swing],
    index: pd.DatetimeIndex,
    forward_only: bool = True,
) -> pd.Series:
    """Extend a two-point line across `index`.

    Positions are bar counts, not calendar time, so a weekend gap does not
    bend the line.

    `forward_only` starts the line at the first anchor. Extending it
    backwards is mathematically fine and visually useless: a steep line run
    back across months leaves the chart, dragging the y-axis with it.
    """
    a, b = anchors
    pos = {ts: i for i, ts in enumerate(index)}
    if a.ts not in pos or b.ts not in pos:
        return pd.Series(np.nan, index=index, dtype="float64")

    x0, x1 = pos[a.ts], pos[b.ts]
    if x1 == x0:
        return pd.Series(np.nan, index=index, dtype="float64")

    slope = (b.price - a.price) / (x1 - x0)
    xs = np.arange(len(index), dtype="float64")
    out = pd.Series(a.price + slope * (xs - x0), index=index, dtype="float64")
    if forward_only:
        out[xs < x0] = np.nan
    return out
