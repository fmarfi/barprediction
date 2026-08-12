"""Indicator checks against published reference values and invariants.

Run with:  python -m pytest tests -q      (or just: python tests/test_indicators.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import indicators as ind  # noqa: E402
from core.predictors import sanitize  # noqa: E402

# The StockCharts RSI-14 worked example.
STOCKCHARTS_CLOSE = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]


def _frame(close, high=None, low=None, vol=1000.0) -> pd.DataFrame:
    close = np.asarray(close, dtype="float64")
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close if high is None else np.asarray(high, dtype="float64"),
            "Low": close if low is None else np.asarray(low, dtype="float64"),
            "Close": close,
            "Volume": np.full(len(close), vol),
        },
        index=idx,
    )


def test_rsi_matches_stockcharts():
    """Track the published table, allowing for its rounded intermediates.

    StockCharts rounds average gain/loss at each step in its spreadsheet, so
    its printed values sit a consistent ~0.07 above an unrounded Wilder
    calculation. test_rsi_first_value_is_exact below pins the true number.
    """
    df = _frame(STOCKCHARTS_CLOSE)
    got = ind.rsi(df, 14).dropna().to_numpy()

    assert len(got) == len(STOCKCHARTS_RSI), (len(got), len(STOCKCHARTS_RSI))
    np.testing.assert_allclose(got, STOCKCHARTS_RSI, atol=0.1)


def test_rsi_first_value_is_exact():
    # Hand-computed from Wilder's definition over the first 14 changes:
    # avg gain 3.34/14, avg loss 1.40/14 -> RS 2.3857142857 -> RSI 70.4641350211
    df = _frame(STOCKCHARTS_CLOSE)
    first = ind.rsi(df, 14).dropna().iloc[0]
    assert abs(first - 70.46413502109705) < 1e-9, first


def test_rsi_bounds_and_extremes():
    # Unbroken rally -> RSI pinned at 100, never NaN from divide-by-zero.
    up = _frame(np.arange(1, 40, dtype="float64"))
    r = ind.rsi(up, 14).dropna()
    assert np.isclose(r.iloc[-1], 100.0), r.iloc[-1]

    down = _frame(np.arange(40, 1, -1, dtype="float64"))
    r = ind.rsi(down, 14).dropna()
    assert np.isclose(r.iloc[-1], 0.0), r.iloc[-1]

    noisy = _frame(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 200)))
    r = ind.rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_wilder_rma_seed_is_sma():
    s = pd.Series(np.arange(1, 21, dtype="float64"))
    got = ind.rma(s, 5)
    # First defined value sits at index 4 and equals mean(1..5) = 3.
    assert got.isna().iloc[:4].all()
    assert np.isclose(got.iloc[4], 3.0), got.iloc[4]


def test_stochastic_formula():
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    df = _frame(close, high, low)

    st = ind.stochastic(df, k=5, smooth_k=1, d=3)
    k = st["%K(5,1)"]
    # Steadily rising: close sits 1 below the window high, range = 5+1+1 = 7,
    # so %K = 100 * (C - (C-4-1)) / ((C+1) - (C-4-1)) = 100 * 5/6.
    assert np.isclose(k.iloc[-1], 100.0 * 5.0 / 6.0), k.iloc[-1]
    assert k.dropna().between(0, 100).all()


def test_stochastic_flat_range_does_not_explode():
    df = _frame([50.0] * 30)
    st = ind.stochastic(df, k=14, smooth_k=3, d=3)
    for name, s in st.items():
        assert not np.isinf(s.to_numpy(dtype="float64")).any(), name


def test_price_oscillator_signs():
    rising = _frame(100 * np.exp(np.linspace(0, 0.5, 120)))
    po = ind.price_oscillator(rising, 12, 26, 9)
    ppo = po["PPO(12,26)"].dropna()
    # Fast EMA leads a rising series -> PPO positive.
    assert (ppo > 0).all(), ppo.min()

    falling = _frame(100 * np.exp(np.linspace(0, -0.5, 120)))
    ppo = ind.price_oscillator(falling, 12, 26, 9)["PPO(12,26)"].dropna()
    assert (ppo < 0).all(), ppo.max()


def test_parabolic_sar_position_and_flip():
    # Rise then fall, so the SAR must flip once and sit on the right side.
    up = np.linspace(100, 150, 40)
    down = np.linspace(150, 100, 40)
    close = np.concatenate([up, down])
    df = _frame(close, close + 0.5, close - 0.5)

    res = ind.parabolic_sar(df)
    sar, trend = res["SAR"], res["trend"]

    valid = sar.notna()
    rising = valid & (trend == 1)
    falling = valid & (trend == -1)

    assert rising.sum() > 0 and falling.sum() > 0, "expected both phases"
    # Allow the flip bar itself, where SAR resets to the prior extreme point.
    assert (sar[rising] <= df["High"][rising]).all()
    assert (sar[falling] >= df["Low"][falling]).all()

    flips = (trend.diff() != 0) & trend.ne(0) & sar.notna()
    assert flips.sum() >= 1


def test_parabolic_sar_af_is_capped():
    close = np.linspace(100, 400, 200)
    df = _frame(close, close + 1, close - 1)
    sar = ind.parabolic_sar(df, af0=0.02, step=0.02, af_max=0.20)["SAR"]
    # With the cap honoured the stop trails price rather than overtaking it.
    assert (sar.dropna() < df["High"].loc[sar.dropna().index]).all()


def test_qqe_shape_and_trend():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = _frame(close, close + 1, close - 1)

    q = ind.qqe(df)
    rsi_ma = q["QQE RSI MA(14,5)"].dropna()
    trail = q["QQE trailing"].dropna()
    trend = q["trend"]

    assert rsi_ma.between(0, 100).all()
    assert len(trail) > 0
    assert set(trend.unique()) <= {1, -1}
    # The trailing stop brackets the oscillator on the correct side.
    common = rsi_ma.index.intersection(trail.index)
    up = common[trend.loc[common] == 1]
    dn = common[trend.loc[common] == -1]
    assert (trail.loc[up] <= rsi_ma.loc[up] + 1e-9).all()
    assert (trail.loc[dn] >= rsi_ma.loc[dn] - 1e-9).all()


def test_qqe_trailing_is_monotonic_within_a_leg():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = _frame(close, close + 1, close - 1)
    q = ind.qqe(df)
    trail, trend = q["QQE trailing"], q["trend"]

    # Inside an uptrend leg the long band may never fall.
    valid = trail.notna()
    t = trend[valid].to_numpy()
    v = trail[valid].to_numpy()
    for i in range(1, len(v)):
        if t[i] == 1 and t[i - 1] == 1:
            assert v[i] >= v[i - 1] - 1e-9, f"long band fell at {i}"
        if t[i] == -1 and t[i - 1] == -1:
            assert v[i] <= v[i - 1] + 1e-9, f"short band rose at {i}"


def test_moving_averages():
    s = pd.Series(np.arange(1, 11, dtype="float64"))
    assert np.isclose(ind.sma(s, 5).iloc[-1], 8.0)
    assert np.isclose(ind.wma(s, 5).iloc[-1], (6 + 14 + 24 + 36 + 50) / 15)
    assert ind.ema(s, 5).isna().sum() == 4


def test_sanitize_fixes_inverted_bars():
    bad = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [5.0],   # high below the body
            "Low": [12.0],   # low above the body
            "Close": [11.0],
            "Volume": [-5.0],
        }
    )
    out = sanitize(bad)
    assert out["High"].iloc[0] == 12.0
    assert out["Low"].iloc[0] == 5.0
    assert out["Volume"].iloc[0] == 0.0
    assert out["High"].iloc[0] >= out["Low"].iloc[0]


def _main() -> int:
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
