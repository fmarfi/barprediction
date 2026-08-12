"""Market data loading for BIST instruments via yfinance."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Interval -> the pandas offset used to extend the index into the future.
# BIST trades Mon-Fri, so daily and slower bars step on business days.
_FUTURE_FREQ = {
    "1d": "B",
    "1wk": "W-FRI",
    "1mo": "BME",
    "60m": "60min",
    "30m": "30min",
    "15m": "15min",
    "5m": "5min",
}

# yfinance refuses long lookbacks on intraday intervals.
MAX_PERIOD = {
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
}


@dataclass(frozen=True)
class Series:
    """A loaded price series plus the metadata the rest of the app needs."""

    symbol: str
    interval: str
    df: pd.DataFrame

    @property
    def future_freq(self) -> str:
        return _FUTURE_FREQ.get(self.interval, "B")

    @property
    def last_bar(self) -> pd.Series:
        return self.df.iloc[-1]


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (field, ticker); reduce to field."""
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def _drop_partial_bar(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop a still-forming final bar.

    Intraday and same-session daily rows come back with zero volume and
    open==high==low==close. Leaving them in drags every indicator toward a
    flat value and makes the last SAR/QQE flip look real when it is not.
    """
    if df.empty:
        return df
    last = df.iloc[-1]
    flat = last["High"] == last["Low"] == last["Close"] == last["Open"]
    if last["Volume"] == 0 and flat:
        return df.iloc[:-1]
    return df


def load(
    symbol: str, period: str = "2y", interval: str = "1d", retries: int = 3
) -> Series:
    """Fetch a symbol from Yahoo. BIST tickers carry the .IS suffix.

    Yahoo rate-limits bursts of requests and answers with an empty frame
    rather than an error, so an empty result is retried with backoff before
    being reported as a missing symbol.
    """
    symbol = symbol.strip().upper()
    if interval in MAX_PERIOD:
        period = MAX_PERIOD[interval]

    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        except Exception:  # noqa: BLE001 - network/parse errors are retryable
            raw = None
        if raw is not None and not raw.empty:
            break
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for {symbol!r} after {retries} attempts. "
            f"Check the ticker (BIST needs the .IS suffix, e.g. THYAO.IS), "
            f"or wait a moment -- Yahoo rate-limits frequent requests."
        )

    df = _flatten(raw, symbol)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    df = df[OHLCV].astype("float64")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = _drop_partial_bar(df, interval)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return Series(symbol=symbol, interval=interval, df=df)


def load_csv(file, symbol: str = "CSV", interval: str = "1d") -> Series:
    """Load a CSV with Date/Open/High/Low/Close[/Volume] columns."""
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in ("date", "datetime", "time") if k in cols), None)
    if date_col is None:
        raise ValueError("CSV needs a Date (or Datetime/Time) column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    rename = {}
    for want in OHLCV:
        src = cols.get(want.lower())
        if src is not None:
            rename[src] = want
    df = df.rename(columns=rename)

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}")

    df = df[OHLCV].astype("float64").dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return Series(symbol=symbol, interval=interval, df=df)


def future_index(series: Series, horizon: int) -> pd.DatetimeIndex:
    """Timestamps for the next `horizon` bars after the end of history."""
    last = series.df.index[-1]
    idx = pd.date_range(
        start=last, periods=horizon + 1, freq=series.future_freq, inclusive="right"
    )
    return pd.DatetimeIndex(idx)
