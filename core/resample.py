"""Convert hand-drawn bars between timeframes.

Draw a scenario on weekly bars, switch the chart to daily, and the weekly
shape is expanded into daily bars that still add up to what you drew:

    first daily open  == the weekly open
    last daily close  == the weekly close
    max of daily highs == the weekly high
    min of daily lows  == the weekly low

The closes in between follow a straight line from open to close rather than
wandering, so filling in the gaps never invents swings you did not ask for.
Going the other way (daily -> weekly) is ordinary OHLC aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Size of one bar in units of the family's smallest bar. Trading bars, not
# calendar time: a week is five sessions, a month roughly twenty-one.
BAR_UNITS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "60m": 12,
    "1d": 1,
    "1wk": 5,
    "1mo": 21,
}

FAMILY = {
    "5m": "intraday",
    "15m": "intraday",
    "30m": "intraday",
    "60m": "intraday",
    "1d": "daily",
    "1wk": "daily",
    "1mo": "daily",
}


class IntervalMismatch(ValueError):
    """Raised when two intervals cannot be related to each other."""


def ratio(src: str, dst: str) -> float:
    """How many `dst` bars fit in one `src` bar."""
    if src not in BAR_UNITS or dst not in BAR_UNITS:
        raise IntervalMismatch(f"unsupported interval pair {src!r} -> {dst!r}")
    if FAMILY[src] != FAMILY[dst]:
        raise IntervalMismatch(
            f"cannot convert {src} to {dst}: intraday and daily bars have no "
            f"fixed ratio (it depends on session length)."
        )
    return BAR_UNITS[src] / BAR_UNITS[dst]


def _split_bar(bar: pd.Series, k: int) -> list[dict]:
    """Expand one coarse bar into `k` finer bars that reproduce its OHLC."""
    o, h, l, c = (float(bar[x]) for x in ("Open", "High", "Low", "Close"))
    vol = float(bar["Volume"]) / k

    # Straight-line path of closes, landing exactly on the coarse close.
    closes = [o + (c - o) * (i + 1) / k for i in range(k)]
    opens = [o] + closes[:-1]

    highs = [max(a, b) for a, b in zip(opens, closes)]
    lows = [min(a, b) for a, b in zip(opens, closes)]

    # Park the extremes at the ends, in the order a real bar of this
    # direction usually makes them: an up bar dips first and peaks last.
    if c >= o:
        lows[0] = min(lows[0], l)
        highs[-1] = max(highs[-1], h)
    else:
        highs[0] = max(highs[0], h)
        lows[-1] = min(lows[-1], l)

    return [
        {
            "Open": opens[i],
            "High": highs[i],
            "Low": lows[i],
            "Close": closes[i],
            "Volume": vol,
        }
        for i in range(k)
    ]


def upsample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """One coarse bar becomes `k` finer bars."""
    rows: list[dict] = []
    for _, bar in df.iterrows():
        rows.extend(_split_bar(bar, k))
    return pd.DataFrame(rows, columns=COLUMNS).astype("float64")


def downsample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """`k` fine bars collapse into one coarse bar. A short tail still counts."""
    rows: list[dict] = []
    for start in range(0, len(df), k):
        chunk = df.iloc[start : start + k]
        if chunk.empty:
            continue
        rows.append(
            {
                "Open": float(chunk["Open"].iloc[0]),
                "High": float(chunk["High"].max()),
                "Low": float(chunk["Low"].min()),
                "Close": float(chunk["Close"].iloc[-1]),
                "Volume": float(chunk["Volume"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS).astype("float64")


def convert(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """Re-express bars drawn at `src` in terms of `dst` bars.

    The returned frame has a plain RangeIndex; the caller assigns real
    timestamps, since only it knows where the scenario should start.
    """
    if df.empty:
        return df.copy()

    r = ratio(src, dst)
    out = df[COLUMNS].astype("float64").reset_index(drop=True)

    if r == 1:
        return out
    if r > 1:
        return upsample(out, int(round(r)))
    return downsample(out, int(round(1 / r)))


def describe(df: pd.DataFrame, src: str, dst: str) -> str:
    """Human-readable summary of what a conversion will do."""
    try:
        r = ratio(src, dst)
    except IntervalMismatch as e:
        return str(e)
    n = len(df)
    if r == 1:
        return f"{n} bar(s), no conversion needed."
    if r > 1:
        return (
            f"{n} {src} bar(s) -> {int(n * r)} {dst} bars, gaps filled along a "
            f"straight line between your open and close."
        )
    k = int(round(1 / r))
    return f"{n} {src} bar(s) -> {int(np.ceil(n / k))} {dst} bar(s), aggregated."
