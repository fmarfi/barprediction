"""Swing detection, impulse legs, Fibonacci levels and trend lines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import trends  # noqa: E402


def _frame(close, high=None, low=None) -> pd.DataFrame:
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5 if high is None else np.asarray(high, "float64"),
            "Low": close - 0.5 if low is None else np.asarray(low, "float64"),
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        },
        index=pd.bdate_range("2026-01-01", periods=len(close)),
    )


# A clean V: down to a trough at index 10, then up to a peak at index 20.
V = _frame(
    list(np.linspace(120, 100, 11)) + list(np.linspace(102, 140, 10))
)


def test_finds_the_obvious_trough_and_peak():
    pts = trends.swing_points(V, 3, 3)
    kinds = {p.kind for p in pts}
    assert kinds, "no pivots found at all"
    lows = [p for p in pts if p.kind == "low"]
    assert lows, "the trough was missed"
    assert abs(min(p.price for p in lows) - float(V["Low"].min())) < 1e-6


def test_no_pivots_on_a_straight_line():
    flat = _frame(np.linspace(100, 200, 60))
    # A pure ramp has no local extremum in the interior.
    assert trends.swing_points(flat, 3, 3) == []


def test_confirmation_bars_are_respected():
    """A pivot cannot exist inside the last `right` bars."""
    pts = trends.swing_points(V, 3, 3)
    if pts:
        assert pts[-1].ts <= V.index[len(V) - 4]


def test_swing_points_survive_degenerate_input():
    assert trends.swing_points(_frame([1.0, 2.0]), 3, 3) == []
    assert trends.swing_points(_frame([]), 3, 3) == []
    assert trends.swing_points(V, 0, 3) == []


def test_last_impulse_runs_between_opposite_pivots():
    imp = trends.last_impulse(V, 2, 2)
    assert imp is not None
    assert imp.start_ts < imp.end_ts
    assert imp.up, "the V should end on an up leg"
    assert imp.size > 0


def test_impulse_direction_on_an_inverted_v():
    inv = _frame(list(np.linspace(100, 140, 11)) + list(np.linspace(138, 100, 10)))
    imp = trends.last_impulse(inv, 2, 2)
    assert imp is not None
    assert not imp.up
    assert imp.direction == "down"


def test_fib_retracements_span_the_leg():
    imp = trends.Impulse(pd.Timestamp("2026-01-01"), 100.0,
                         pd.Timestamp("2026-02-01"), 200.0)
    lv = trends.fib_levels(imp)
    # 0 is the end of the move, 1 is where it started.
    assert np.isclose(lv["0"], 200.0)
    assert np.isclose(lv["1"], 100.0)
    assert np.isclose(lv["0.5"], 150.0)
    assert np.isclose(lv["0.618"], 200.0 - 100.0 * 0.618)
    # Every retracement sits inside the leg.
    for k, v in lv.items():
        assert 100.0 - 1e-9 <= v <= 200.0 + 1e-9, (k, v)


def test_fib_projections_continue_past_the_leg():
    imp = trends.Impulse(pd.Timestamp("2026-01-01"), 100.0,
                         pd.Timestamp("2026-02-01"), 200.0)
    lv = trends.fib_levels(imp, retracements=(), projections=(1.618, 2.618))
    assert np.isclose(lv["1.618"], 261.8)
    assert np.isclose(lv["2.618"], 361.8)


def test_fib_on_a_down_leg_mirrors():
    imp = trends.Impulse(pd.Timestamp("2026-01-01"), 200.0,
                         pd.Timestamp("2026-02-01"), 100.0)
    lv = trends.fib_levels(imp, projections=(1.618,))
    assert np.isclose(lv["0"], 100.0)
    assert np.isclose(lv["1"], 200.0)
    assert np.isclose(lv["0.618"], 100.0 + 100.0 * 0.618)
    # A down leg projects downward.
    assert lv["1.618"] < 100.0
    assert np.isclose(lv["1.618"], 38.2)


def test_fib_labels_are_tidy():
    imp = trends.Impulse(pd.Timestamp("2026-01-01"), 0.0,
                         pd.Timestamp("2026-02-01"), 10.0)
    keys = set(trends.fib_levels(imp, projections=(1.618,)))
    assert "0" in keys and "1" in keys and "0.618" in keys and "1.618" in keys
    assert not any(k.endswith(".") or k.endswith("0") and "." in k for k in keys)


def test_trend_line_projects_through_its_anchors():
    anchors = trends.trend_line(V, 2, 2)
    if anchors is None:
        return  # not every synthetic shape yields two same-kind pivots
    a, b = anchors
    line = trends.project(anchors, V.index)
    assert np.isclose(line.loc[a.ts], a.price)
    assert np.isclose(line.loc[b.ts], b.price)
    assert line.notna().all()


def test_project_is_linear_in_bar_position_not_calendar_time():
    a = trends.Swing(V.index[0], 100.0, "low")
    b = trends.Swing(V.index[10], 110.0, "low")
    line = trends.project((a, b), V.index)
    # One unit per bar, regardless of the weekend gaps in the index.
    steps = np.diff(line.to_numpy())
    assert np.allclose(steps, steps[0])
    assert np.isclose(steps[0], 1.0)


def test_project_handles_anchors_outside_the_index():
    a = trends.Swing(pd.Timestamp("1999-01-01"), 1.0, "low")
    b = trends.Swing(V.index[5], 2.0, "low")
    assert trends.project((a, b), V.index).isna().all()


def test_manual_impulse_picks_the_drawn_direction():
    up = trends.manual_impulse(V, 10, 20)
    assert up is not None and up.up
    down = trends.manual_impulse(V, 0, 10)
    assert down is not None and not down.up
    assert trends.manual_impulse(V, 5, 5) is None


def test_impulse_pct():
    imp = trends.Impulse(pd.Timestamp("2026-01-01"), 100.0,
                         pd.Timestamp("2026-02-01"), 125.0)
    assert np.isclose(imp.pct, 25.0)


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
