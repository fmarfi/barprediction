"""Interval conversion must preserve the shape the user drew.

The contract for coarse -> fine: the finer bars have to add back up to the
coarse bar they came from, and must not invent swings between the open and
the close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import resample, store  # noqa: E402

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _bars(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLS).astype("float64")


UP = _bars([[100, 112, 96, 110, 1000], [110, 125, 108, 120, 2000]])
DOWN = _bars([[120, 122, 100, 105, 1500], [105, 108, 90, 95, 1200]])


def test_weekly_to_daily_reproduces_each_weekly_bar():
    for src in (UP, DOWN):
        out = resample.convert(src, "1wk", "1d")
        assert len(out) == len(src) * 5

        for i in range(len(src)):
            week = src.iloc[i]
            days = out.iloc[i * 5 : (i + 1) * 5]
            assert np.isclose(days["Open"].iloc[0], week["Open"]), "open"
            assert np.isclose(days["Close"].iloc[-1], week["Close"]), "close"
            assert np.isclose(days["High"].max(), week["High"]), "high"
            assert np.isclose(days["Low"].min(), week["Low"]), "low"
            assert np.isclose(days["Volume"].sum(), week["Volume"]), "volume"


def test_upsampled_closes_are_monotonic():
    """No invented fluctuation: closes walk straight from open to close."""
    for src in (UP, DOWN):
        out = resample.convert(src, "1wk", "1d")
        for i in range(len(src)):
            week = src.iloc[i]
            closes = out["Close"].iloc[i * 5 : (i + 1) * 5].to_numpy()
            steps = np.diff(closes)
            if week["Close"] >= week["Open"]:
                assert (steps >= -1e-9).all(), f"up week dipped: {closes}"
            else:
                assert (steps <= 1e-9).all(), f"down week rose: {closes}"


def test_upsampled_bars_are_internally_valid():
    out = resample.convert(UP, "1wk", "1d")
    assert (out["High"] >= out["Open"]).all()
    assert (out["High"] >= out["Close"]).all()
    assert (out["Low"] <= out["Open"]).all()
    assert (out["Low"] <= out["Close"]).all()
    assert (out["High"] >= out["Low"]).all()


def test_daily_to_weekly_aggregates():
    daily = _bars([[100 + i, 105 + i, 95 + i, 102 + i, 100] for i in range(10)])
    out = resample.convert(daily, "1d", "1wk")
    assert len(out) == 2
    first = daily.iloc[:5]
    assert np.isclose(out["Open"].iloc[0], first["Open"].iloc[0])
    assert np.isclose(out["Close"].iloc[0], first["Close"].iloc[-1])
    assert np.isclose(out["High"].iloc[0], first["High"].max())
    assert np.isclose(out["Low"].iloc[0], first["Low"].min())
    assert np.isclose(out["Volume"].iloc[0], first["Volume"].sum())


def test_daily_to_weekly_keeps_a_partial_final_week():
    daily = _bars([[100, 101, 99, 100, 10] for _ in range(7)])
    out = resample.convert(daily, "1d", "1wk")
    assert len(out) == 2, "a 2-day remainder must still produce a bar"


def test_round_trip_weekly_daily_weekly_is_identity():
    for src in (UP, DOWN):
        back = resample.convert(resample.convert(src, "1wk", "1d"), "1d", "1wk")
        for c in COLS:
            np.testing.assert_allclose(back[c], src[c], atol=1e-9, err_msg=c)


def test_same_interval_is_a_passthrough():
    out = resample.convert(UP, "1d", "1d")
    np.testing.assert_allclose(out[COLS].to_numpy(), UP[COLS].to_numpy())


def test_monthly_to_daily_ratio():
    out = resample.convert(UP.iloc[:1], "1mo", "1d")
    assert len(out) == 21


def test_cross_family_conversion_is_refused():
    for a, b in (("1d", "60m"), ("15m", "1wk")):
        try:
            resample.convert(UP, a, b)
        except resample.IntervalMismatch as e:
            assert "no fixed ratio" in str(e)
        else:
            raise AssertionError(f"{a}->{b} should have been refused")


def test_intraday_conversions():
    out = resample.convert(UP, "60m", "15m")
    assert len(out) == len(UP) * 4
    back = resample.convert(out, "15m", "60m")
    np.testing.assert_allclose(back["Close"], UP["Close"], atol=1e-9)


def test_empty_input_survives():
    empty = _bars([])
    assert resample.convert(empty, "1wk", "1d").empty


def test_store_round_trip(tmp_dir: Path | None = None):
    idx = pd.bdate_range("2026-08-13", periods=2)
    bars = UP.copy()
    bars.index = idx

    name = "__pytest_scenario__"
    try:
        store.save(name, "TEST.IS", "1wk", bars)
        got = store.load(name)
        assert got.symbol == "TEST.IS"
        assert got.interval == "1wk"
        assert len(got.bars) == 2
        np.testing.assert_allclose(got.bars["Close"], bars["Close"])
        assert any(s.name == name for s in store.list_all())
    finally:
        store.delete(name)
    assert not any(s.name == name for s in store.list_all())


def test_download_upload_round_trip():
    """Bytes handed to the browser must parse back to the same scenario."""
    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)

    blob = store.to_json_bytes("my scenario", "THYAO.IS", "1wk", bars)
    sc = store.from_json_bytes(blob)

    assert sc.name == "my scenario"
    assert sc.symbol == "THYAO.IS"
    assert sc.interval == "1wk"
    for c in COLS:
        np.testing.assert_allclose(sc.bars[c], bars[c], atol=1e-9, err_msg=c)
    assert list(sc.bars.index) == list(bars.index)


def test_uploaded_weekly_scenario_converts_to_daily():
    """The hosted-app path: download weekly, re-upload onto a daily chart."""
    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)
    sc = store.from_json_bytes(store.to_json_bytes("w", "X", "1wk", bars))

    daily = resample.convert(sc.bars, sc.interval, "1d")
    assert len(daily) == 10
    for i in range(2):
        wk, dd = bars.iloc[i], daily.iloc[i * 5 : (i + 1) * 5]
        assert np.isclose(dd["Open"].iloc[0], wk["Open"])
        assert np.isclose(dd["Close"].iloc[-1], wk["Close"])
        assert np.isclose(dd["High"].max(), wk["High"])
        assert np.isclose(dd["Low"].min(), wk["Low"])


def test_upload_rejects_junk_with_a_readable_message():
    for blob, why in (
        (b"not json at all", "plain text"),
        (b"[1,2,3]", "a list, not an object"),
        (b'{"hello": 1}', "no bars key"),
        (b'{"bars": []}', "empty bars"),
        (b'{"bars": [{"ts": "2026-01-01", "Open": 1}]}', "missing OHLC columns"),
    ):
        try:
            store.from_json_bytes(blob)
        except ValueError as e:
            assert str(e), f"{why}: empty error message"
        else:
            raise AssertionError(f"should have rejected {why}")


def test_filename_is_filesystem_safe():
    name = store.filename_for("my / bad: name*", "XU100.IS", "1d")
    assert not set(name) & set('/\\:*?"<>|')
    assert name.endswith(".json")


def test_store_rejects_blank_name_and_empty_bars():
    for name, bars, why in (
        ("", UP, "blank name"),
        ("ok", _bars([]), "empty bars"),
    ):
        try:
            store.save(name, "T", "1d", bars)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {why}")


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
