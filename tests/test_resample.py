"""Interval conversion must preserve the shape the user drew.

The contract for coarse -> fine: the finer bars have to add back up to the
coarse bar they came from, and must not invent swings between the open and
the close.
"""

from __future__ import annotations

import json
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


def test_random_fill_preserves_every_coarse_bar():
    for src in (UP, DOWN):
        out = resample.convert(src, "1wk", "1d", fill=resample.RANDOM, seed=3)
        assert len(out) == len(src) * 5
        for i in range(len(src)):
            week, days = src.iloc[i], out.iloc[i * 5 : (i + 1) * 5]
            assert np.isclose(days["Open"].iloc[0], week["Open"]), "open"
            assert np.isclose(days["Close"].iloc[-1], week["Close"]), "close"
            assert np.isclose(days["High"].max(), week["High"]), "high"
            assert np.isclose(days["Low"].min(), week["Low"]), "low"


def test_random_fill_stays_inside_the_coarse_range():
    for src in (UP, DOWN):
        out = resample.convert(src, "1wk", "1d", fill=resample.RANDOM, seed=5)
        for i in range(len(src)):
            week, days = src.iloc[i], out.iloc[i * 5 : (i + 1) * 5]
            assert (days["High"] <= week["High"] + 1e-9).all()
            assert (days["Low"] >= week["Low"] - 1e-9).all()


def test_random_fill_is_reproducible_and_seed_sensitive():
    a = resample.convert(UP, "1wk", "1d", fill=resample.RANDOM, seed=11)
    b = resample.convert(UP, "1wk", "1d", fill=resample.RANDOM, seed=11)
    c = resample.convert(UP, "1wk", "1d", fill=resample.RANDOM, seed=12)
    # Stability matters: Streamlit reruns constantly and the chart must not
    # reshuffle itself whenever an unrelated widget moves.
    assert resample.same_ohlc(a, b)
    assert not resample.same_ohlc(a, c)


def test_random_fill_does_not_flatten_against_the_bounds():
    """Scaling, not clipping: a clipped bridge leaves runs of dead closes."""
    wide = _bars([[100, 125, 92, 118, 1000], [118, 140, 112, 120, 900]])
    out = resample.convert(wide, "1wk", "1d", fill=resample.RANDOM, seed=7)
    flat = int((out["Close"].diff().abs() < 1e-9).sum())
    assert flat == 0, f"{flat} consecutive identical closes"


def test_random_fill_holds_invariants_over_many_shapes():
    rng = np.random.default_rng(0)
    for trial in range(150):
        o = float(rng.uniform(50, 500))
        c = o * float(rng.uniform(0.85, 1.15))
        h = max(o, c) * float(rng.uniform(1.0, 1.15))
        l = min(o, c) * float(rng.uniform(0.85, 1.0))
        src = _bars([[o, h, l, c, 100]])
        for a, b in (("1wk", "1d"), ("1mo", "1d"), ("60m", "5m")):
            out = resample.convert(src, a, b, fill=resample.RANDOM, seed=trial)
            assert np.isclose(out["Open"].iloc[0], o), (trial, a, b)
            assert np.isclose(out["Close"].iloc[-1], c), (trial, a, b)
            assert np.isclose(out["High"].max(), h), (trial, a, b)
            assert np.isclose(out["Low"].min(), l), (trial, a, b)
            assert (out["High"] >= out[["Open", "Close"]].max(axis=1) - 1e-9).all()
            assert (out["Low"] <= out[["Open", "Close"]].min(axis=1) + 1e-9).all()


def test_random_fill_round_trips_back_to_the_coarse_bar():
    out = resample.convert(UP, "1wk", "1d", fill=resample.RANDOM, seed=2)
    back = resample.convert(out, "1d", "1wk")
    for c in ("Open", "High", "Low", "Close"):
        np.testing.assert_allclose(back[c], UP[c], atol=1e-9, err_msg=c)


def test_straight_fill_is_still_the_monotonic_one():
    out = resample.convert(UP, "1wk", "1d", fill=resample.STRAIGHT, seed=0)
    for i in range(len(UP)):
        closes = out["Close"].iloc[i * 5 : (i + 1) * 5].to_numpy()
        assert (np.diff(closes) >= -1e-9).all()


def test_downsampling_loses_detail_that_upsampling_cannot_invent():
    """Documents why the app memoises bars per interval.

    One weekly bar cannot encode five distinct daily paths, so
    daily -> weekly -> daily is lossy in a way the reverse is not. The
    dashboard works around it by remembering what you drew at each
    timeframe; this test pins the underlying maths.
    """
    daily = _bars(
        [
            [100, 104, 99, 103, 10],
            [103, 109, 102, 108, 10],
            [108, 110, 101, 102, 10],
            [102, 106, 100, 105, 10],
            [105, 112, 104, 111, 10],
        ]
    )
    weekly = resample.convert(daily, "1d", "1wk")
    back = resample.convert(weekly, "1wk", "1d")

    # The envelope survives...
    assert np.isclose(back["Open"].iloc[0], daily["Open"].iloc[0])
    assert np.isclose(back["Close"].iloc[-1], daily["Close"].iloc[-1])
    assert np.isclose(back["High"].max(), daily["High"].max())
    assert np.isclose(back["Low"].min(), daily["Low"].min())
    # ...but the interior path does not, and must not be claimed to.
    assert not resample.same_ohlc(back, daily)


def test_same_ohlc():
    a = _bars([[100, 110, 90, 105, 10], [105, 115, 100, 108, 20]])
    assert resample.same_ohlc(a, a.copy())

    # Volume and index are deliberately ignored.
    b = a.copy()
    b["Volume"] = [999.0, 888.0]
    b.index = pd.bdate_range("2030-01-01", periods=2)
    assert resample.same_ohlc(a, b)

    c = a.copy()
    c.loc[c.index[1], "Close"] = 108.5
    assert not resample.same_ohlc(c, a)

    assert not resample.same_ohlc(a, a.iloc[:1])
    assert not resample.same_ohlc(a, None)
    assert not resample.same_ohlc(None, a)


def test_same_ohlc_tolerance():
    a = _bars([[100, 110, 90, 105, 10]])
    b = a.copy()
    b.loc[b.index[0], "Close"] = 105 + 1e-9
    assert resample.same_ohlc(a, b)
    b.loc[b.index[0], "Close"] = 105.01
    assert not resample.same_ohlc(a, b)


def test_fit_length_truncates_keeping_the_front():
    src = _bars([[100 + i, 105 + i, 95 + i, 102 + i, 10] for i in range(8)])
    out = resample.fit_length(src, 3)
    assert len(out) == 3
    np.testing.assert_allclose(out["Close"].to_numpy(), src["Close"].to_numpy()[:3])


def test_fit_length_pads_flat_from_the_last_close():
    src = _bars([[100, 110, 90, 105, 10], [105, 115, 100, 108, 20]])
    out = resample.fit_length(src, 5)
    assert len(out) == 5
    # Drawn bars survive untouched.
    np.testing.assert_allclose(out["Close"].to_numpy()[:2], [105.0, 108.0])
    # Padding is flat at the last close, so it adds no direction.
    pad = out.iloc[2:]
    for c in ("Open", "High", "Low", "Close"):
        np.testing.assert_allclose(pad[c].to_numpy(), [108.0] * 3, err_msg=c)


def test_fit_length_edge_cases():
    src = _bars([[100, 110, 90, 105, 10]])
    assert len(resample.fit_length(src, 1)) == 1
    assert resample.fit_length(src, 0).empty
    assert resample.fit_length(src, -2).empty
    assert resample.fit_length(_bars([]), 4).empty


def test_shrinking_then_growing_keeps_the_surviving_bars():
    """Nudging the horizon down and back must not corrupt what is left."""
    src = _bars([[100 + i, 106 + i, 94 + i, 103 + i, 10] for i in range(6)])
    small = resample.fit_length(src, 2)
    back = resample.fit_length(small, 6)
    np.testing.assert_allclose(back["Close"].to_numpy()[:2], src["Close"].to_numpy()[:2])
    assert len(back) == 6


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


def test_notes_survive_a_save_and_load():
    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)
    notes = [
        {"ts": "2026-08-13", "price": 110.5, "text": "gap fills here"},
        {"ts": "2026-08-14", "price": 96.0, "text": "watch this level"},
    ]
    sc = store.from_json_bytes(
        store.to_json_bytes("n", "X", "1d", bars, notes)
    )
    assert len(sc.notes) == 2
    assert sc.notes[0]["text"] == "gap fills here"
    assert np.isclose(sc.notes[1]["price"], 96.0)


def test_malformed_notes_are_dropped_not_raised():
    """One bad note must not take the download button -- and the page -- down."""
    bad = [
        {"ts": "2026-08-13", "price": 100.0, "text": "good"},
        {"ts": "not-a-date", "price": 1.0, "text": "bad date"},
        {"ts": "2026-08-14", "price": "x", "text": "bad price"},
        {"nope": 1},
        None,
    ]
    cleaned = store.clean_notes(bad)
    assert len(cleaned) == 1 and cleaned[0]["text"] == "good"

    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)
    sc = store.from_json_bytes(store.to_json_bytes("n", "X", "1d", bars, bad))
    assert len(sc.notes) == 1


def test_note_offsets_round_trip_and_default():
    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)
    notes = [
        {"ts": "2026-08-13", "price": 110.0, "text": "nudged", "dx": 40, "dy": -60},
        {"ts": "2026-08-14", "price": 96.0, "text": "default placement"},
    ]
    sc = store.from_json_bytes(store.to_json_bytes("n", "X", "1d", bars, notes))
    assert np.isclose(sc.notes[0]["dx"], 40.0)
    assert np.isclose(sc.notes[0]["dy"], -60.0)
    # An unpositioned note sits just above what it points at.
    assert np.isclose(sc.notes[1]["dx"], 0.0)
    assert np.isclose(sc.notes[1]["dy"], -34.0)


def test_note_offset_zero_is_kept_not_defaulted():
    """dy=0 is a real position and must not be replaced by the default."""
    cleaned = store.clean_notes(
        [{"ts": "2026-08-13", "price": 1.0, "text": "flat", "dx": 0, "dy": 0}]
    )
    assert np.isclose(cleaned[0]["dy"], 0.0)


def test_files_without_notes_still_load():
    bars = UP.copy()
    bars.index = pd.bdate_range("2026-08-13", periods=2)
    blob = store.to_json_bytes("n", "X", "1d", bars)
    payload = json.loads(blob.decode())
    del payload["notes"]  # as written before notes existed
    sc = store.from_json_bytes(json.dumps(payload).encode())
    assert sc.notes == ()


def test_ip_is_local_accepts_loopback():
    for ip in ("127.0.0.1", "::1", "localhost", "LOCALHOST", " 127.0.0.1 ",
               "0:0:0:0:0:0:0:1", "::ffff:127.0.0.1"):
        assert store.ip_is_local(ip), ip


def test_ip_is_local_fails_closed():
    """Anything not clearly loopback must count as remote.

    Guessing wrong here shows one person's saved scenarios to another, so
    LAN addresses, None, and stand-in objects from test harnesses are all
    treated as remote.
    """
    class Mock:
        pass

    for ip in ("10.30.4.53", "192.168.1.7", "212.174.147.219", "0.0.0.0",
               "", None, 127, Mock(), b"127.0.0.1"):
        assert not store.ip_is_local(ip), repr(ip)


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
