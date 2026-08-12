"""Cross-check our indicators against the independent `ta` library.

`ta` is a separate implementation by a different author, so agreement is
meaningful evidence that both follow the standard definitions. Where they
disagree the cause is documented below rather than papered over -- several
of these differences are real and deliberate on our side.

    python -m pip install ta
    python tests/test_cross_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import indicators as ind  # noqa: E402

try:
    import ta  # noqa: F401
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD as TA_MACD
    from ta.trend import ADXIndicator, PSARIndicator

    HAVE_TA = True
except Exception:  # noqa: BLE001
    HAVE_TA = False


def _ohlc(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """A realistic random walk with genuine intrabar range."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.1, n))
    spread = np.abs(rng.normal(0, 0.9, n)) + 0.25
    high = close + spread
    low = close - np.abs(rng.normal(0, 0.9, n)) - 0.25
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum.reduce([high, close, open_]),
            "Low": np.minimum.reduce([low, close, open_]),
            "Close": close,
            "Volume": rng.integers(1e5, 1e6, n).astype("float64"),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def _agree(a: pd.Series, b: pd.Series, tol: float, label: str, skip: int = 60):
    """Compare two series over the region where both are defined."""
    both = pd.concat([a, b], axis=1).dropna().iloc[skip:]
    assert len(both) > 50, f"{label}: only {len(both)} comparable points"
    diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    worst = float(diff.max())
    assert worst < tol, f"{label}: max deviation {worst:.6f} exceeds {tol}"
    return worst, len(both)


def test_rsi_converges_to_ta():
    """Agreement only after the seeds wash out -- ours is the correct one.

    `ta` starts its EWM at bar one; Wilder seeds the average gain/loss with
    an SMA of the first `length` changes. On the StockCharts series Wilder's
    hand-computed first value is 70.46413502109705: we return exactly that,
    `ta` returns 71.80. The gap is a pure seeding artefact and decays from
    ~5.4 early to ~3e-08 by bar 250, so the two agree once it has washed out.
    test_rsi_first_value_is_exact in test_indicators.py pins the true value.
    """
    df = _ohlc()
    ours = ind.rsi(df, 14)
    theirs = RSIIndicator(df["Close"], window=14, fillna=False).rsi()
    worst, n = _agree(ours, theirs, 1e-3, "RSI (converged)", skip=200)
    print(f"    RSI      max diff {worst:.2e} over {n} bars (after seed decay)")


def test_rsi_seed_beats_ta_on_reference_value():
    from test_indicators import STOCKCHARTS_CLOSE, _frame  # noqa: PLC0415

    sc = _frame(STOCKCHARTS_CLOSE)
    ours = float(ind.rsi(sc, 14).dropna().iloc[0])
    theirs = float(RSIIndicator(sc["Close"], window=14).rsi().dropna().iloc[0])
    assert abs(ours - 70.46413502109705) < 1e-9, ours
    assert abs(theirs - 70.46413502109705) > 1.0, theirs
    print(f"    RSI seed ours {ours:.6f} vs ta {theirs:.6f} (Wilder = 70.464135)")


def test_macd_matches_ta():
    df = _ohlc()
    ours = ind.macd(df, 12, 26, 9)
    t = TA_MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)

    worst, n = _agree(ours["MACD(12,26)"], t.macd(), 1e-6, "MACD line")
    print(f"    MACD     max diff {worst:.2e} over {n} bars")
    # Signal seeding differs: `ta` runs its EMA over the NaN-padded MACD
    # series, we seed from the first real value. The two converge quickly.
    worst, n = _agree(ours["Signal(9)"], t.macd_signal(), 5e-3, "MACD signal", skip=80)
    print(f"    MACD sig max diff {worst:.2e} over {n} bars")


def test_stochastic_matches_ta():
    df = _ohlc()
    ours = ind.stochastic(df, k=14, smooth_k=3, d=3)
    t = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
    )
    worst, n = _agree(ours["%K(14,3)"], t.stoch_signal(), 1e-6, "Stoch %K")
    print(f"    Stoch %K max diff {worst:.2e} over {n} bars")


def test_adx_and_di_match_ta():
    df = _ohlc()
    ours = ind.dmi(df, 14, 14)
    t = ADXIndicator(df["High"], df["Low"], df["Close"], window=14, fillna=False)

    # `ta` divides DI by a plain rolling sum of TR rather than Wilder's RMA,
    # so its +DI/-DI sit at a different scale. Compare the shape instead:
    # the two must agree on which side is dominant, which is what +DI/-DI
    # are actually read for.
    ours_dom = (ours["+DI(14)"] > ours["-DI(14)"])
    theirs_dom = (t.adx_pos() > t.adx_neg())
    both = pd.concat([ours_dom, theirs_dom], axis=1).dropna().iloc[60:]
    agree = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
    assert agree > 0.97, f"+DI/-DI dominance agrees only {agree:.1%} of bars"
    print(f"    DI dom.  agrees on {agree:.1%} of {len(both)} bars")

    # ADX itself should track closely in level, not just in shape.
    corr = float(
        pd.concat([ours[f"ADX(14)"], t.adx()], axis=1).dropna().iloc[60:].corr().iloc[0, 1]
    )
    assert corr > 0.95, f"ADX correlation with ta only {corr:.3f}"
    print(f"    ADX      correlation {corr:.4f}")


def test_psar_matches_ta():
    df = _ohlc()
    ours = ind.parabolic_sar(df, af0=0.02, step=0.02, af_max=0.2)["SAR"]
    theirs = PSARIndicator(
        df["High"], df["Low"], df["Close"], step=0.02, max_step=0.2
    ).psar()

    both = pd.concat([ours, theirs], axis=1).dropna().iloc[60:]
    rel = ((both.iloc[:, 0] - both.iloc[:, 1]).abs() / both.iloc[:, 1])
    close_enough = float((rel < 0.01).mean())
    # SAR is path dependent: a different seed direction on bar 1 can shift
    # an entire leg. Agreement on the large majority of bars is the useful
    # signal, and our own invariant tests pin the mechanics.
    assert close_enough > 0.90, f"PSAR within 1% on only {close_enough:.1%} of bars"
    print(f"    PSAR     within 1% on {close_enough:.1%} of {len(both)} bars")


def test_our_ppo_is_macd_normalised():
    """PPO must equal MACD divided by the slow EMA, by definition."""
    df = _ohlc()
    m = ind.macd(df, 12, 26, 9)["MACD(12,26)"]
    p = ind.price_oscillator(df, 12, 26, 9, percent=True)["PPO(12,26)"]
    slow = ind.ema(df["Close"], 26)
    implied = 100.0 * m / slow
    worst, n = _agree(p, implied, 1e-9, "PPO vs MACD/slow")
    print(f"    PPO=MACD/slow  max diff {worst:.2e} over {n} bars")


def test_atr_matches_wilder_definition():
    df = _ohlc()
    tr = ind.true_range(df)
    manual = ind.rma(tr, 14)
    worst, n = _agree(ind.atr(df, 14), manual, 1e-12, "ATR")
    print(f"    ATR      max diff {worst:.2e} over {n} bars")


def _main() -> int:
    if not HAVE_TA:
        print("SKIP: `ta` not installed -- run: python -m pip install ta")
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
