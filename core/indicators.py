"""Technical indicators, implemented directly from their definitions.

Every function takes the OHLCV frame and returns either a Series or a dict of
named Series ready to plot. Nothing here mutates its input.

Wilder's smoothing (RMA) is used wherever the original author specified it --
RSI and QQE -- because an ordinary EMA of the same period gives visibly
different values and would not match a TradingView or MetaTrader chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# smoothing primitives
# --------------------------------------------------------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype="float64")
    return s.rolling(n, min_periods=n).apply(
        lambda x: float(np.dot(x, w) / w.sum()), raw=True
    )


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average: SMA seed, then recursive 1/n smoothing."""
    s = s.astype("float64")
    vals = np.asarray(s.to_numpy(), dtype="float64")
    valid = np.flatnonzero(~np.isnan(vals))
    if valid.size < n:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    start = valid[n - 1]
    acc = float(np.nanmean(vals[valid[0] : start + 1][-n:]))
    # Own the buffer: pandas 3 hands back read-only views under copy-on-write.
    res = np.full(len(vals), np.nan, dtype="float64")
    res[start] = acc
    alpha = 1.0 / n
    for i in range(start + 1, len(vals)):
        v = vals[i]
        if np.isnan(v):
            res[i] = acc
            continue
        acc = acc + alpha * (v - acc)
        res[i] = acc
    return pd.Series(res, index=s.index, dtype="float64")


_MA_FUNCS = {"SMA": sma, "EMA": ema, "WMA": wma, "RMA": rma}
MA_KINDS = tuple(_MA_FUNCS)


def moving_average(s: pd.Series, n: int, kind: str = "EMA") -> pd.Series:
    try:
        return _MA_FUNCS[kind.upper()](s, n)
    except KeyError:
        raise ValueError(f"unknown MA kind {kind!r}; use one of {MA_KINDS}") from None


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where a and b cross in either direction (Pine's ta.cross)."""
    d = a - b
    prev = d.shift(1)
    return (np.sign(d) != np.sign(prev)) & d.notna() & prev.notna()


# --------------------------------------------------------------------------
# RSI
# --------------------------------------------------------------------------


def rsi(df: pd.DataFrame, length: int = 14, source: str = "Close") -> pd.Series:
    delta = df[source].astype("float64").diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of up-closes -> RSI pins at 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain.notna(), np.nan)
    return out.rename(f"RSI({length})")


# --------------------------------------------------------------------------
# Stochastic oscillator
# --------------------------------------------------------------------------


def stochastic(
    df: pd.DataFrame, k: int = 14, smooth_k: int = 3, d: int = 3
) -> dict[str, pd.Series]:
    """Slow stochastic. smooth_k=1 gives the raw (fast) %K."""
    low = df["Low"].rolling(k, min_periods=k).min()
    high = df["High"].rolling(k, min_periods=k).max()
    span = (high - low).replace(0.0, np.nan)

    raw_k = 100.0 * (df["Close"] - low) / span
    # A flat range means no position within it; carry the prior reading.
    raw_k = raw_k.ffill()

    pct_k = sma(raw_k, smooth_k) if smooth_k > 1 else raw_k
    pct_d = sma(pct_k, d)
    return {f"%K({k},{smooth_k})": pct_k, f"%D({d})": pct_d}


# --------------------------------------------------------------------------
# Price Oscillator
# --------------------------------------------------------------------------


def price_oscillator(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    kind: str = "EMA",
    percent: bool = True,
    source: str = "Close",
) -> dict[str, pd.Series]:
    """Percentage Price Oscillator (percent=True) or absolute PO."""
    src = df[source].astype("float64")
    f = moving_average(src, fast, kind)
    s = moving_average(src, slow, kind)

    po = 100.0 * (f - s) / s.replace(0.0, np.nan) if percent else (f - s)
    sig = moving_average(po.dropna(), signal, kind).reindex(po.index)

    label = "PPO" if percent else "PO"
    return {
        f"{label}({fast},{slow})": po,
        f"Signal({signal})": sig,
        "Histogram": po - sig,
    }


# --------------------------------------------------------------------------
# MACD
# --------------------------------------------------------------------------


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    source: str = "Close",
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    The absolute-difference sibling of the Price Oscillator: MACD is in price
    units, PPO expresses the same spread as a percentage. PPO is the one to
    use when comparing symbols at different price levels.
    """
    src = df[source].astype("float64")
    line = ema(src, fast) - ema(src, slow)
    # Signal is seeded from the first defined MACD value, not from the NaNs
    # ahead of it, which is why the EMA runs on the dropped series.
    sig = ema(line.dropna(), signal).reindex(line.index)
    return {
        f"MACD({fast},{slow})": line,
        f"Signal({signal})": sig,
        "Histogram": line - sig,
    }


# --------------------------------------------------------------------------
# Directional movement: +DI, -DI, ADX
# --------------------------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length).rename(f"ATR({length})")


def dmi(
    df: pd.DataFrame, length: int = 14, adx_length: int = 14
) -> dict[str, pd.Series]:
    """Wilder's Directional Movement Index: +DI, -DI and ADX.

    Directional movement counts only the larger of the two range expansions:
    a bar that extends further up than down contributes +DM and no -DM, and
    an inside bar contributes neither.
    """
    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = rma(true_range(df), length)
    # A zero ATR means a run of identical bars; leave DI undefined rather
    # than dividing by zero.
    safe_tr = tr.replace(0.0, np.nan)

    plus_di = 100.0 * rma(plus_dm, length) / safe_tr
    minus_di = 100.0 * rma(minus_dm, length) / safe_tr

    total = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / total.replace(0.0, np.nan)
    adx = rma(dx, adx_length)

    return {
        f"+DI({length})": plus_di,
        f"-DI({length})": minus_di,
        f"ADX({adx_length})": adx,
    }


# --------------------------------------------------------------------------
# Parabolic SAR
# --------------------------------------------------------------------------


def parabolic_sar(
    df: pd.DataFrame, af0: float = 0.02, step: float = 0.02, af_max: float = 0.20
) -> dict[str, pd.Series]:
    """Wilder's Parabolic SAR.

    Returns the stop value plus the trend direction, so the caller can colour
    rising and falling phases separately and detect flips.
    """
    high = df["High"].to_numpy(dtype="float64")
    low = df["Low"].to_numpy(dtype="float64")
    n = len(df)

    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype="int8")
    if n < 2:
        return {
            "SAR": pd.Series(sar, index=df.index),
            "trend": pd.Series(trend, index=df.index),
        }

    # Seed: assume the first leg follows the first bar-to-bar move.
    up = high[1] >= high[0]
    af = af0
    ep = high[1] if up else low[1]
    sar[1] = low[0] if up else high[0]
    trend[1] = 1 if up else -1

    for i in range(2, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)

        if up:
            # The stop may never enter the two prior bars' range.
            cur = min(cur, low[i - 1], low[i - 2])
            if low[i] < cur:  # flip to downtrend
                up = False
                cur = ep
                ep = low[i]
                af = af0
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, af_max)
        else:
            cur = max(cur, high[i - 1], high[i - 2])
            if high[i] > cur:  # flip to uptrend
                up = True
                cur = ep
                ep = high[i]
                af = af0
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, af_max)

        sar[i] = cur
        trend[i] = 1 if up else -1

    return {
        "SAR": pd.Series(sar, index=df.index, name="SAR"),
        "trend": pd.Series(trend, index=df.index, name="SAR trend"),
    }


# --------------------------------------------------------------------------
# QQE
# --------------------------------------------------------------------------


def qqe(
    df: pd.DataFrame,
    rsi_length: int = 14,
    smoothing: int = 5,
    factor: float = 4.238,
    source: str = "Close",
) -> dict[str, pd.Series]:
    """Qualitative Quantitative Estimation.

    A smoothed RSI plus a volatility-scaled trailing stop on that RSI. The
    trailing line only ever moves in the direction of the trend, exactly like
    a SAR applied to the oscillator instead of to price.
    """
    base = rsi(df, rsi_length, source=source)
    rsi_ma = ema(base, smoothing)

    wilders = rsi_length * 2 - 1
    atr_rsi = (rsi_ma.shift(1) - rsi_ma).abs()
    ma_atr_rsi = ema(atr_rsi, wilders)
    dar = ema(ma_atr_rsi, wilders) * factor

    vals = rsi_ma.to_numpy(dtype="float64")
    d = dar.to_numpy(dtype="float64")
    n = len(df)

    longband = np.full(n, np.nan)
    shortband = np.full(n, np.nan)
    trend = np.ones(n, dtype="int8")

    prev_long = 0.0
    prev_short = 0.0
    prev_trend = 1
    started = False

    for i in range(n):
        v, dv = vals[i], d[i]
        if np.isnan(v) or np.isnan(dv):
            continue

        new_long = v - dv
        new_short = v + dv
        pv = vals[i - 1] if i > 0 else np.nan

        if not started:
            prev_long, prev_short = new_long, new_short
            started = True

        if not np.isnan(pv) and pv > prev_long and v > prev_long:
            cur_long = max(prev_long, new_long)
        else:
            cur_long = new_long

        if not np.isnan(pv) and pv < prev_short and v < prev_short:
            cur_short = min(prev_short, new_short)
        else:
            cur_short = new_short

        # Trend flips when the smoothed RSI crosses the opposite band.
        if not np.isnan(pv):
            crossed_up = (pv <= prev_short) != (v <= prev_short)
            crossed_dn = (prev_long <= pv) != (prev_long <= v)
        else:
            crossed_up = crossed_dn = False

        if crossed_up:
            cur_trend = 1
        elif crossed_dn:
            cur_trend = -1
        else:
            cur_trend = prev_trend

        longband[i] = cur_long
        shortband[i] = cur_short
        trend[i] = cur_trend
        prev_long, prev_short, prev_trend = cur_long, cur_short, cur_trend

    trail = np.where(trend == 1, longband, shortband)
    return {
        f"QQE RSI MA({rsi_length},{smoothing})": rsi_ma,
        "QQE trailing": pd.Series(trail, index=df.index, dtype="float64"),
        "trend": pd.Series(trend, index=df.index),
    }


# --------------------------------------------------------------------------
# registry -- the dashboard builds its controls from this
# --------------------------------------------------------------------------

#: name -> (function, overlays_price?, default params)
REGISTRY: dict[str, dict] = {
    "Parabolic SAR": {
        "fn": parabolic_sar,
        "overlay": True,
        "params": {"af0": 0.02, "step": 0.02, "af_max": 0.20},
    },
    "Moving Average": {
        "fn": None,  # handled specially: user can stack several
        "overlay": True,
        "params": {"n": 50, "kind": "EMA"},
    },
    "RSI": {
        "fn": rsi,
        "overlay": False,
        "params": {"length": 14},
    },
    "MACD": {
        "fn": macd,
        "overlay": False,
        "params": {"fast": 12, "slow": 26, "signal": 9},
    },
    "DMI / ADX": {
        "fn": dmi,
        "overlay": False,
        "params": {"length": 14, "adx_length": 14},
    },
    "Stochastic": {
        "fn": stochastic,
        "overlay": False,
        "params": {"k": 14, "smooth_k": 3, "d": 3},
    },
    "Price Oscillator": {
        "fn": price_oscillator,
        "overlay": False,
        "params": {"fast": 12, "slow": 26, "signal": 9, "percent": True},
    },
    "QQE": {
        "fn": qqe,
        "overlay": False,
        "params": {"rsi_length": 14, "smoothing": 5, "factor": 4.238},
    },
}
