"""Detect indicator events that occur inside the predicted region.

This is the question the tool exists to answer: given the bars you drew, does
anything actually trigger? Each rule returns zero or more Event rows keyed to
the bar where the condition first becomes true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    when: pd.Timestamp
    indicator: str
    event: str
    detail: str
    direction: str  # bullish | bearish | neutral


def _crosses(s: pd.Series, level: float) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for upward and downward crossings of a fixed level."""
    prev = s.shift(1)
    up = (prev <= level) & (s > level)
    dn = (prev >= level) & (s < level)
    return up.fillna(False), dn.fillna(False)


def _cross_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Masks for a crossing above / below b."""
    d = a - b
    prev = d.shift(1)
    up = (prev <= 0) & (d > 0)
    dn = (prev >= 0) & (d < 0)
    return up.fillna(False), dn.fillna(False)


def _emit(
    mask: pd.Series,
    window: pd.DatetimeIndex,
    indicator: str,
    event: str,
    direction: str,
    series: pd.Series | None = None,
) -> list[Event]:
    out: list[Event] = []
    hits = window.intersection(mask.index[mask.fillna(False)])
    for ts in hits:
        val = ""
        if series is not None and ts in series.index:
            v = series.loc[ts]
            if pd.notna(v):
                val = f"value {float(v):.2f}"
        out.append(Event(ts, indicator, event, val, direction))
    return out


def collect(
    active: dict[str, dict[str, pd.Series]],
    close: pd.Series,
    window: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Scan every active indicator for events landing in `window`."""
    events: list[Event] = []

    for name, series_map in active.items():
        if name == "Parabolic SAR":
            trend = series_map.get("trend")
            if trend is not None:
                flip = trend.diff().ne(0) & trend.ne(0) & trend.shift(1).ne(0)
                up = flip & trend.eq(1)
                dn = flip & trend.eq(-1)
                events += _emit(up, window, "Parabolic SAR", "flip to long", "bullish",
                                series_map.get("SAR"))
                events += _emit(dn, window, "Parabolic SAR", "flip to short", "bearish",
                                series_map.get("SAR"))

        elif name == "RSI":
            s = next(iter(series_map.values()))
            u70, d70 = _crosses(s, 70)
            u30, d30 = _crosses(s, 30)
            u50, d50 = _crosses(s, 50)
            events += _emit(u70, window, "RSI", "crossed above 70 (overbought)", "bearish", s)
            events += _emit(d70, window, "RSI", "dropped back below 70", "bearish", s)
            events += _emit(d30, window, "RSI", "crossed below 30 (oversold)", "bullish", s)
            events += _emit(u30, window, "RSI", "recovered above 30", "bullish", s)
            events += _emit(u50, window, "RSI", "crossed above 50", "bullish", s)
            events += _emit(d50, window, "RSI", "crossed below 50", "bearish", s)

        elif name == "Stochastic":
            keys = list(series_map)
            k, d = series_map[keys[0]], series_map[keys[1]]
            up, dn = _cross_pair(k, d)
            events += _emit(up, window, "Stochastic", "%K crossed above %D", "bullish", k)
            events += _emit(dn, window, "Stochastic", "%K crossed below %D", "bearish", k)
            u80, _ = _crosses(k, 80)
            _, d20 = _crosses(k, 20)
            events += _emit(u80, window, "Stochastic", "%K above 80 (overbought)", "bearish", k)
            events += _emit(d20, window, "Stochastic", "%K below 20 (oversold)", "bullish", k)

        elif name == "Price Oscillator":
            keys = list(series_map)
            po, sig = series_map[keys[0]], series_map[keys[1]]
            up, dn = _cross_pair(po, sig)
            events += _emit(up, window, "Price Oscillator", "crossed above signal", "bullish", po)
            events += _emit(dn, window, "Price Oscillator", "crossed below signal", "bearish", po)
            uz, dz = _crosses(po, 0.0)
            events += _emit(uz, window, "Price Oscillator", "crossed above zero", "bullish", po)
            events += _emit(dz, window, "Price Oscillator", "crossed below zero", "bearish", po)

        elif name == "MACD":
            keys = list(series_map)
            line, sig = series_map[keys[0]], series_map[keys[1]]
            up, dn = _cross_pair(line, sig)
            events += _emit(up, window, "MACD", "crossed above signal", "bullish", line)
            events += _emit(dn, window, "MACD", "crossed below signal", "bearish", line)
            uz, dz = _crosses(line, 0.0)
            events += _emit(uz, window, "MACD", "crossed above zero", "bullish", line)
            events += _emit(dz, window, "MACD", "crossed below zero", "bearish", line)

        elif name == "DMI / ADX":
            keys = list(series_map)
            plus, minus, adx = (series_map[k] for k in keys[:3])
            up, dn = _cross_pair(plus, minus)
            events += _emit(up, window, "DMI / ADX", "+DI crossed above -DI",
                            "bullish", plus)
            events += _emit(dn, window, "DMI / ADX", "+DI crossed below -DI",
                            "bearish", minus)
            u25, d25 = _crosses(adx, 25.0)
            events += _emit(u25, window, "DMI / ADX",
                            "ADX rose above 25 (trend strengthening)", "neutral", adx)
            events += _emit(d25, window, "DMI / ADX",
                            "ADX fell below 25 (trend fading)", "neutral", adx)

        elif name == "QQE":
            trend = series_map.get("trend")
            rsi_ma = next(iter(series_map.values()))
            if trend is not None:
                flip = trend.diff().ne(0) & trend.ne(0) & trend.shift(1).ne(0)
                events += _emit(flip & trend.eq(1), window, "QQE", "trend flipped long",
                                "bullish", rsi_ma)
                events += _emit(flip & trend.eq(-1), window, "QQE", "trend flipped short",
                                "bearish", rsi_ma)
            u50, d50 = _crosses(rsi_ma, 50)
            events += _emit(u50, window, "QQE", "RSI-MA crossed above 50", "bullish", rsi_ma)
            events += _emit(d50, window, "QQE", "RSI-MA crossed below 50", "bearish", rsi_ma)

        elif name == "Moving Average":
            mas = list(series_map.items())
            for label, s in mas:
                up, dn = _cross_pair(close, s)
                events += _emit(up, window, "Moving Average",
                                f"price crossed above {label}", "bullish", s)
                events += _emit(dn, window, "Moving Average",
                                f"price crossed below {label}", "bearish", s)
            # Fast/slow crossovers between the stacked averages.
            for i in range(len(mas)):
                for j in range(i + 1, len(mas)):
                    (la, sa), (lb, sb) = mas[i], mas[j]
                    up, dn = _cross_pair(sa, sb)
                    events += _emit(up, window, "Moving Average",
                                    f"{la} crossed above {lb}", "bullish", sa)
                    events += _emit(dn, window, "Moving Average",
                                    f"{la} crossed below {lb}", "bearish", sa)

    cols = ["_ts", "Bar", "Indicator", "Event", "Detail", "Bias"]
    if not events:
        return pd.DataFrame(columns=cols)

    # Include the time component only when bars are intraday, otherwise
    # several 5-minute events would all render as the same date.
    intraday = any(e.when.time() != pd.Timestamp(0).time() for e in events)
    fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"

    rows = [
        {
            # Kept for chart markers; the dashboard drops it before display.
            "_ts": e.when,
            "Bar": e.when.strftime(fmt),
            "Indicator": e.indicator,
            "Event": e.event,
            "Detail": e.detail,
            "Bias": e.direction,
        }
        for e in sorted(events, key=lambda e: (e.when, e.indicator))
    ]
    return pd.DataFrame(rows, columns=cols)


def snapshot(
    active: dict[str, dict[str, pd.Series]],
    at_history: pd.Timestamp,
    at_forecast: pd.Timestamp | None,
) -> pd.DataFrame:
    """Side-by-side indicator readings: last real bar vs last predicted bar."""
    rows = []
    for name, series_map in active.items():
        for label, s in series_map.items():
            if label == "trend":
                continue
            now = s.loc[at_history] if at_history in s.index else np.nan
            then = (
                s.loc[at_forecast]
                if at_forecast is not None and at_forecast in s.index
                else np.nan
            )
            delta = (
                then - now if pd.notna(now) and pd.notna(then) else np.nan
            )
            rows.append(
                {
                    "Indicator": name,
                    "Series": label,
                    "Now": None if pd.isna(now) else round(float(now), 3),
                    "After scenario": None if pd.isna(then) else round(float(then), 3),
                    "Change": None if pd.isna(delta) else round(float(delta), 3),
                }
            )
    return pd.DataFrame(rows)
