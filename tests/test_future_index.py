"""future_index must hand back exactly the number of bars requested.

Regression guard: the original implementation leaned on
`date_range(..., inclusive="right")`, which only drops the leading stamp when
it coincides with the anchor. Daily bars anchor on a business day so it
worked; a mid-week anchor on a weekly frequency produced one bar too many,
and a 3-week scenario silently expanded into 20 daily bars instead of 15.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data  # noqa: E402


def _series(last: str, interval: str) -> data.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(last)])
    df = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=idx,
    )
    return data.Series(symbol="T", interval=interval, df=df)


def test_length_is_exact_for_every_interval_and_anchor():
    # Every weekday as an anchor, so no frequency can hide an alignment bug.
    anchors = [
        "2026-08-10",  # Monday
        "2026-08-11",
        "2026-08-12",  # Wednesday
        "2026-08-13",
        "2026-08-14",  # Friday
    ]
    for interval in ("1d", "1wk", "1mo", "60m", "15m"):
        for a in anchors:
            for h in (1, 2, 3, 5, 12, 30):
                got = data.future_index(_series(a, interval), h)
                assert len(got) == h, (
                    f"{interval} anchored {a} horizon {h} -> {len(got)} stamps"
                )


def test_all_stamps_are_after_the_anchor_and_ascending():
    for interval in ("1d", "1wk", "1mo"):
        s = _series("2026-08-12", interval)
        idx = data.future_index(s, 8)
        assert (idx > s.df.index[-1]).all()
        assert idx.is_monotonic_increasing
        assert idx.is_unique


def test_daily_index_skips_weekends():
    idx = data.future_index(_series("2026-08-14", "1d"), 4)  # Friday anchor
    assert [d.strftime("%a") for d in idx] == ["Mon", "Tue", "Wed", "Thu"]


def test_zero_and_negative_horizon_give_nothing():
    s = _series("2026-08-12", "1d")
    assert len(data.future_index(s, 0)) == 0
    assert len(data.future_index(s, -3)) == 0


def test_three_weekly_bars_expand_to_fifteen_daily():
    """The exact case that surfaced the bug."""
    from core import predictors, resample  # noqa: PLC0415

    s = _series("2026-08-12", "1wk")  # a Wednesday
    idx = data.future_index(s, 3)
    assert len(idx) == 3, f"expected 3 weekly stamps, got {len(idx)}"

    weekly = predictors.build("Manual - flat").propose(s.df, idx)
    daily = resample.convert(weekly, "1wk", "1d")
    assert len(daily) == 15, f"expected 15 daily bars, got {len(daily)}"


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
