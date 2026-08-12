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


STRAIGHT = "straight"
RANDOM = "random"
FILL_MODES = (RANDOM, STRAIGHT)


def _close_path(o: float, c: float, h: float, l: float, k: int, rng) -> np.ndarray:
    """Closes for `k` sub-bars, running from just after `o` to exactly `c`.

    With no generator this is a straight line. With one it is a Brownian
    bridge -- a random walk pinned at both ends -- so the interior wanders
    like a real session instead of ruling a line, while still landing on the
    close you drew. The path is confined to [l, h], so no sub-bar can escape
    the range of the coarse bar it came from.
    """
    t = np.arange(1, k + 1, dtype="float64") / k
    line = o + (c - o) * t
    if rng is None or k < 2:
        return line

    span = h - l
    if span <= 0:
        return line

    # Pin a random walk at both ends, then scale it to the bar's range.
    steps = rng.normal(0.0, 1.0, k)
    walk = np.cumsum(steps)
    bridge = walk - t * walk[-1]
    peak = float(np.abs(bridge).max())
    if peak < 1e-9:
        return line

    # Scale so the path fits inside [l, h] on its own. Clipping instead
    # would flatten every excursion against the boundary and leave runs of
    # identical closes, which look nothing like a real session.
    headroom = np.where(bridge >= 0, h - line, line - l)
    fits = float(np.min(headroom / np.maximum(np.abs(bridge), 1e-9)))
    scale = min(0.32 * span / peak, max(fits, 0.0) * 0.95)

    path = line + scale * bridge
    path[-1] = c  # the close you drew is not negotiable
    return np.clip(path, l, h)


def _split_bar(bar: pd.Series, k: int, rng=None) -> list[dict]:
    """Expand one coarse bar into `k` finer bars that reproduce its OHLC."""
    o, h, l, c = (float(bar[x]) for x in ("Open", "High", "Low", "Close"))
    vol = float(bar["Volume"]) / k

    closes = _close_path(o, c, h, l, k, rng)
    opens = np.r_[o, closes[:-1]]

    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)

    if rng is not None and k > 1:
        # Give each sub-bar its own wick, kept inside the coarse range.
        span = h - l
        highs = np.minimum(highs + np.abs(rng.normal(0, 0.10 * span, k)), h)
        lows = np.maximum(lows - np.abs(rng.normal(0, 0.10 * span, k)), l)
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

    # The coarse extremes have to appear somewhere, or the aggregate no
    # longer matches what was drawn. Put them on the bars already nearest.
    highs[int(np.argmax(highs))] = h
    lows[int(np.argmin(lows))] = l

    if rng is None:
        # Straight-line mode keeps the classic placement: an up bar dips
        # first and peaks last.
        highs = np.maximum(opens, closes)
        lows = np.minimum(opens, closes)
        if c >= o:
            lows[0] = min(lows[0], l)
            highs[-1] = max(highs[-1], h)
        else:
            highs[0] = max(highs[0], h)
            lows[-1] = min(lows[-1], l)

    return [
        {
            "Open": float(opens[i]),
            "High": float(highs[i]),
            "Low": float(lows[i]),
            "Close": float(closes[i]),
            "Volume": vol,
        }
        for i in range(k)
    ]


def upsample(
    df: pd.DataFrame, k: int, fill: str = STRAIGHT, seed: int = 0
) -> pd.DataFrame:
    """One coarse bar becomes `k` finer bars.

    `fill` picks how the interior is drawn: STRAIGHT rules a line from open
    to close, RANDOM wanders between them. RANDOM is seeded per bar index,
    so the result is stable across Streamlit reruns -- the chart must not
    reshuffle itself every time an unrelated widget moves.
    """
    rows: list[dict] = []
    for i, (_, bar) in enumerate(df.iterrows()):
        rng = (
            np.random.default_rng([int(seed), i]) if fill == RANDOM else None
        )
        rows.extend(_split_bar(bar, k, rng))
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


def convert(
    df: pd.DataFrame,
    src: str,
    dst: str,
    fill: str = STRAIGHT,
    seed: int = 0,
) -> pd.DataFrame:
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
        return upsample(out, int(round(r)), fill=fill, seed=seed)
    return downsample(out, int(round(1 / r)))


def same_ohlc(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-6) -> bool:
    """Do two frames hold the same bars, ignoring index and volume?"""
    if a is None or b is None or len(a) != len(b):
        return False
    cols = ["Open", "High", "Low", "Close"]
    try:
        return bool(
            np.allclose(
                a[cols].to_numpy(dtype="float64"),
                b[cols].to_numpy(dtype="float64"),
                atol=tol,
                rtol=0,
            )
        )
    except Exception:  # noqa: BLE001 - shape or dtype mismatch means "no"
        return False


def fit_length(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Make the frame exactly `n` bars long without discarding drawn work.

    Shrinking keeps the leading bars; growing continues flat from the last
    close, which adds no direction of its own. Used when the horizon slider
    moves, so nudging it no longer wipes a scenario.
    """
    if n <= 0:
        return df.iloc[:0].copy()
    if df.empty:
        return df.copy()
    if len(df) == n:
        return df.copy()
    if len(df) > n:
        return df.iloc[:n].copy()

    last = df.iloc[-1]
    close = float(last["Close"])
    pad = pd.DataFrame(
        [
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": float(last["Volume"]),
            }
        ]
        * (n - len(df)),
        columns=COLUMNS,
    )
    return pd.concat([df[COLUMNS], pad], ignore_index=True).astype("float64")


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
