"""BIST bar-scenario dashboard.

Pick a symbol, propose the next few bars (draw them yourself or let a source
generate them), then switch indicators on and off to see what your scenario
does to them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import (
    charting,
    data,
    indicators as ind,
    predictors,
    resample,
    signals,
    store,
    trends,
)

st.set_page_config(
    page_title="BIST Bar Prediction", layout="wide", initial_sidebar_state="expanded"
)

# A starting watchlist; any Yahoo ticker works in the free-text box.
PRESETS = [
    "XU100.IS", "THYAO.IS", "ASELS.IS", "GARAN.IS", "AKBNK.IS", "ISCTR.IS",
    "KCHOL.IS", "SAHOL.IS", "EREGL.IS", "TUPRS.IS", "BIMAS.IS", "SISE.IS",
    "FROTO.IS", "TCELL.IS", "PGSUS.IS", "SASA.IS", "HEKTS.IS", "KOZAL.IS",
]
INTERVALS = ["1d", "1wk", "1mo", "60m", "30m", "15m", "5m"]
PERIODS = ["6mo", "1y", "2y", "5y", "10y", "max"]

# Upsampling multiplies bars (a month becomes 21 days), so the ceiling has
# to leave room for a converted scenario rather than truncating it.
MAX_HORIZON = 250

INDICATORS = [
    "Trend & Fibonacci",
    "Parabolic SAR",
    "Moving averages",
    "RSI",
    "MACD",
    "DMI / ADX",
    "Stochastic",
    "Price Oscillator",
    "QQE",
]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      div[data-testid="stMetricLabel"] {opacity: .75;}
      section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        border: none; margin-top: -.5rem;
      }
      div[data-testid="stElementToolbar"] {opacity: .35;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def _load(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return data.load(symbol, period, interval).df


def _viewer_is_local() -> bool:
    """Is the person looking at this page sitting at the machine serving it?

    Files under `scenarios/` live on the server. Streamlit prints a Network
    URL by default, so without this check anyone on the LAN would get a
    Load/Delete list for someone else's scenarios.

    Two signals, either of which is sufficient: the server bound to loopback
    so nobody else can reach it at all, or this particular client connected
    from loopback.
    """
    if os.environ.get("BARPREDICTION_NO_LOCAL_FILES") == "1":
        return False

    try:
        bound = st.get_option("server.address")
    except Exception:  # noqa: BLE001
        bound = None
    if store.ip_is_local(bound):
        return True

    try:
        return store.ip_is_local(st.context.ip_address)
    except Exception:  # noqa: BLE001 - no context in bare/test runs
        return False


def _show_local_files() -> bool:
    return not store.is_ephemeral() and _viewer_is_local()


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Data")

    source = st.radio("Source", ["Yahoo (BIST)", "CSV upload"], horizontal=True)

    if source == "CSV upload":
        upload = st.file_uploader("OHLC CSV", type=["csv"])
        symbol = st.text_input("Label", "CSV")
        interval = st.selectbox("Bar interval", INTERVALS, index=0)
        period = "max"
    else:
        upload = None
        preset = st.selectbox("Symbol", PRESETS, index=0)
        custom = st.text_input(
            "…or type a ticker", "", placeholder="e.g. TOASO.IS"
        ).strip().upper()
        symbol = custom or preset
        col_a, col_b = st.columns(2)
        interval = col_a.selectbox("Interval", INTERVALS, index=0)
        period = col_b.selectbox("History", PERIODS, index=2)

    # Read before the conversion block below, which runs before the horizon
    # slider and needs these settings already resolved.
    fill_label = st.selectbox(
        "Gap fill on finer intervals",
        ["Random walk", "Straight line"],
        help="Going from a coarse interval to a finer one has to invent the "
        "bars in between. Either way your open, high, low and close are "
        "reproduced exactly — only the path between them differs.",
    )
    fill_mode = resample.RANDOM if fill_label == "Random walk" else resample.STRAIGHT
    fill_seed = (
        st.number_input(
            "Fill seed", 0, 9999, 0, step=1, help="Change it to reroll the path."
        )
        if fill_mode == resample.RANDOM
        else 0
    )

    # Re-rolling has to invalidate remembered bars, or a switch back would
    # restore the previous path and the new seed would look ignored.
    fill_sig = (fill_mode, int(fill_seed))
    prev_fill = st.session_state.get("_fill_sig")
    fill_changed = prev_fill is not None and prev_fill != fill_sig
    if fill_changed:
        st.session_state["bars_memo"] = {}
    st.session_state["_fill_sig"] = fill_sig

    # Anything that changes the bar count has to be resolved here, before the
    # horizon slider is created: Streamlit refuses to let a widget's state be
    # written once the widget exists.
    def _stage(frame, note: str) -> None:
        st.session_state["_loaded_bars"] = frame
        st.session_state["horizon"] = int(min(max(len(frame), 1), MAX_HORIZON))
        if len(frame) > MAX_HORIZON:
            note += f" (trimmed to {MAX_HORIZON})"
        st.session_state["_load_note"] = note

    def _remember_fill_source(src_bars, src_iv: str) -> None:
        """Keep the coarse bars an upsample came from.

        Without them, changing the gap-fill setting could only affect the
        *next* conversion -- the finer bars already on screen would keep the
        path they were born with.
        """
        try:
            upsampled = resample.ratio(src_iv, interval) > 1
        except resample.IntervalMismatch:
            upsampled = False
        if upsampled:
            st.session_state["fill_source"] = {
                "interval": src_iv,
                "bars": src_bars.copy(),
                "target": interval,
            }
        else:
            st.session_state.pop("fill_source", None)

    if st.session_state.get("_pending_scenario") is not None:
        try:
            sc = st.session_state.pop("_pending_scenario")
            conv = resample.convert(
                sc.bars, sc.interval, interval, fill=fill_mode, seed=int(fill_seed)
            )
            if sc.notes:
                st.session_state["notes"] = [dict(n) for n in sc.notes]
            _remember_fill_source(sc.bars, sc.interval)
            _stage(
                conv,
                f"Loaded “{sc.name}” ({sc.symbol} {sc.interval})"
                + (
                    f" → converted to {len(conv)} × {interval} bars"
                    if sc.interval != interval
                    else ""
                ),
            )
        except Exception as e:  # noqa: BLE001
            st.session_state["_load_error"] = f"Could not load that scenario: {e}"
            st.session_state.pop("_pending_scenario", None)

    # Switching timeframe converts the bars you already drew rather than
    # throwing them away: draw on weekly, flip to daily, keep your shape.
    meta = st.session_state.get("bars_meta")
    have = st.session_state.get("bars")
    if (
        meta
        and have is not None
        and not have.empty
        and meta["symbol"] == symbol
        and meta["interval"] != interval
        and "_loaded_bars" not in st.session_state
    ):
        try:
            prev_iv = meta["interval"]
            conv = resample.convert(
                have, prev_iv, interval, fill=fill_mode, seed=int(fill_seed)
            )
            note = (
                f"Converted your {len(have)} × {prev_iv} bars → "
                f"{len(conv)} × {interval}"
            )

            # Coarsening loses detail: five distinct daily bars collapse into
            # one weekly bar, and expanding that back gives a straight line
            # rather than the days you drew. So if we have been at this
            # interval before and the coarse view has not been edited since,
            # restore exactly what was drawn instead of re-deriving it.
            memo = st.session_state.get("bars_memo", {})
            kept = memo.get(interval)
            if kept is not None and resample.same_ohlc(
                resample.convert(
                    kept, interval, prev_iv, fill=fill_mode, seed=int(fill_seed)
                ),
                have,
            ):
                conv = kept
                note = f"Restored your {len(kept)} × {interval} bars"

            _remember_fill_source(have, prev_iv)
            _stage(conv, note)
        except resample.IntervalMismatch as e:
            # Intraday <-> daily has no fixed ratio; reseed rather than guess.
            st.session_state["_load_error"] = f"Kept your bars out of it — {e}"
            st.session_state["_force_reseed"] = True

    # Changing the fill setting re-draws the gaps in the bars already on
    # screen, rather than waiting for the next interval switch. Only when
    # those bars are still exactly what the old setting produced -- if they
    # have been hand-edited since, re-filling would throw the edits away.
    if fill_changed and "_loaded_bars" not in st.session_state:
        fsrc = st.session_state.get("fill_source")
        cur = st.session_state.get("bars")
        if fsrc and cur is not None and fsrc["target"] == interval:
            try:
                as_was = resample.convert(
                    fsrc["bars"], fsrc["interval"], interval,
                    fill=prev_fill[0], seed=prev_fill[1],
                )
                if resample.same_ohlc(as_was, cur):
                    redone = resample.convert(
                        fsrc["bars"], fsrc["interval"], interval,
                        fill=fill_mode, seed=int(fill_seed),
                    )
                    _stage(redone, f"Re-filled the gaps — {fill_label.lower()}")
                else:
                    st.session_state["_load_note"] = (
                        "Kept your edited bars. Switch interval and back to "
                        "re-fill the gaps with this setting."
                    )
            except resample.IntervalMismatch:
                pass

    st.divider()
    st.subheader("Bars to predict")

    if "horizon" not in st.session_state:
        st.session_state["horizon"] = 5
    horizon = st.slider("How many bars", 1, MAX_HORIZON, key="horizon")
    src_name = st.selectbox(
        "Bar source", predictors.names(), index=0,
        help="How the bars are seeded. You can edit any of them afterwards.",
    )

    proto = predictors.REGISTRY[src_name]
    st.caption(proto.description)

    cfg: dict = {}
    for p in proto.params:
        if p.kind == "int":
            cfg[p.key] = st.number_input(
                p.label, int(p.min), int(p.max), int(p.default),
                step=int(p.step or 1), help=p.help or None,
            )
        elif p.kind == "bool":
            cfg[p.key] = st.checkbox(p.label, bool(p.default), help=p.help or None)
        elif p.kind == "choice":
            cfg[p.key] = st.selectbox(p.label, p.choices, help=p.help or None)
        else:
            cfg[p.key] = st.number_input(
                p.label, float(p.min), float(p.max), float(p.default),
                step=float(p.step or 0.1), help=p.help or None,
            )

    regen = st.button("Reseed bars", width="stretch", type="secondary")

    st.divider()
    st.subheader("Indicator settings")
    st.caption("Switch indicators on above the chart; tune them here.")

    with st.expander("Trend & Fibonacci"):
        swing_n = st.slider(
            "Swing sensitivity", 1, 15, 4,
            help="Bars either side a pivot must beat. Higher finds fewer, "
            "bigger swings.",
        )
        fib_ratios = st.multiselect(
            "Retracements",
            [f"{r:g}" for r in trends.RETRACEMENTS],
            default=["0", "0.382", "0.5", "0.618", "1"],
        )
        fib_projs = st.multiselect(
            "Projections",
            [f"{r:g}" for r in trends.PROJECTIONS],
            default=["1.618"],
            help="Targets beyond the end of the leg, in its direction.",
        )
        show_trendline = st.checkbox("Trend line through pivots", True)

    with st.expander("Parabolic SAR"):
        sar_af0 = st.number_input("Step (AF start)", 0.001, 0.5, 0.02, step=0.005, format="%.3f")
        sar_step = st.number_input("Increment", 0.001, 0.5, 0.02, step=0.005, format="%.3f")
        sar_max = st.number_input("Max AF", 0.05, 1.0, 0.20, step=0.05)

    with st.expander("Moving averages"):
        ma_cfg = st.data_editor(
            pd.DataFrame(
                {
                    "On": [True, True, False, False],
                    "Type": ["EMA", "EMA", "SMA", "SMA"],
                    "Period": [20, 50, 100, 200],
                }
            ),
            column_config={
                "On": st.column_config.CheckboxColumn(width="small"),
                "Type": st.column_config.SelectboxColumn(options=list(ind.MA_KINDS)),
                "Period": st.column_config.NumberColumn(min_value=2, max_value=500, step=1),
            },
            hide_index=True,
            width="stretch",
            key="ma_cfg",
        )

    with st.expander("RSI"):
        rsi_len = st.number_input("Length", 2, 100, 14, step=1)

    with st.expander("MACD"):
        macd_fast = st.number_input("Fast", 2, 100, 12, step=1, key="macd_fast")
        macd_slow = st.number_input("Slow", 3, 200, 26, step=1, key="macd_slow")
        macd_sig = st.number_input("Signal", 1, 50, 9, step=1, key="macd_sig")

    with st.expander("DMI / ADX"):
        dmi_len = st.number_input("DI length", 2, 100, 14, step=1)
        adx_len = st.number_input("ADX smoothing", 2, 100, 14, step=1)

    with st.expander("Stochastic"):
        st_k = st.number_input("%K length", 1, 100, 14, step=1)
        st_sk = st.number_input("%K smoothing", 1, 20, 3, step=1)
        st_d = st.number_input("%D length", 1, 20, 3, step=1)

    with st.expander("Price Oscillator"):
        po_fast = st.number_input("Fast", 2, 100, 12, step=1)
        po_slow = st.number_input("Slow", 3, 200, 26, step=1)
        po_sig = st.number_input("Signal", 1, 50, 9, step=1)
        po_kind = st.selectbox("MA type", list(ind.MA_KINDS), index=1)
        po_pct = st.checkbox("Percentage (PPO)", True)

    with st.expander("QQE"):
        qqe_rsi = st.number_input("RSI length", 2, 100, 14, step=1, key="qqe_rsi")
        qqe_sf = st.number_input("Smoothing factor", 1, 50, 5, step=1)
        qqe_f = st.number_input("QQE factor", 0.1, 10.0, 4.238, step=0.1)

    st.divider()
    st.subheader("Display")
    show_tail = st.slider(
        "History bars shown",
        40,
        600,
        150,
        step=10,
        help="Candlesticks are the expensive part of the chart. Keep this "
        "low if panning feels heavy.",
    )
    log_scale = st.checkbox("Log price axis", False)
    crosshair = st.checkbox(
        "Crosshair",
        False,
        help="Spike lines across every panel. Looks good, costs pan/zoom "
        "smoothness on long histories.",
    )
    unified_hover = st.checkbox(
        "Unified hover",
        False,
        help="One tooltip listing every series at the cursor, instead of the "
        "nearest one. Slower with several indicators on.",
    )
    drop_last = st.checkbox(
        "Drop the final bar",
        False,
        help="Mid-session the newest bar is still forming. Tick this to "
        "exclude it so indicators reflect completed bars only.",
    )


# ------------------------------------------------------------------ data

try:
    if source == "CSV upload":
        if upload is None:
            st.info("Upload a CSV with Date, Open, High, Low, Close columns to begin.")
            st.stop()
        hist = data.load_csv(upload, symbol, interval).df
    else:
        with st.spinner(f"Loading {symbol}…"):
            hist = _load(symbol, period, interval)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load {symbol}: {e}")
    st.stop()

if drop_last and len(hist) > 1:
    hist = hist.iloc[:-1]

if len(hist) < 30:
    st.warning(f"Only {len(hist)} bars available; longer indicators will be blank.")

series = data.Series(symbol=symbol, interval=interval, df=hist)
fut_idx = data.future_index(series, horizon)

# ------------------------------------------------- predicted bars (state)

# Bars persist across setting changes. Only an explicit reseed, a new
# symbol, or a change of bar source throws away what you drew -- moving the
# horizon resizes, and changing interval converts (handled in the sidebar).
src_sig = (src_name, tuple(sorted(cfg.items())))
meta = st.session_state.get("bars_meta")

loaded = st.session_state.pop("_loaded_bars", None)
forced = st.session_state.pop("_force_reseed", False)
existing = st.session_state.get("bars")

if loaded is not None:
    base = loaded
elif (
    forced
    or existing is None
    or existing.empty
    or meta is None
    or meta["symbol"] != symbol
    or regen
    or st.session_state.get("_src") != src_sig
):
    base = predictors.build(src_name, **cfg).propose(hist, fut_idx)
else:
    base = existing

# Length follows the horizon slider; the index always re-anchors to the
# current future stamps, so a new trading day shifts bars without wiping.
base = resample.fit_length(base, len(fut_idx))
base = base.reset_index(drop=True)
base.index = fut_idx
base.index.name = "Date"

st.session_state["bars"] = predictors.sanitize(base)
st.session_state["_src"] = src_sig

# Remember this timeframe's bars so switching away and back restores the
# detail rather than a re-derived approximation. A reseed or a new symbol
# invalidates everything remembered.
if (
    meta is None
    or meta["symbol"] != symbol
    or forced
    or regen
    or (loaded is None and st.session_state.get("_src_prev") != src_sig)
):
    st.session_state["bars_memo"] = {}
    # Freshly seeded bars did not come from an upsample, so there is no
    # coarse source left to re-fill from.
    st.session_state.pop("fill_source", None)
st.session_state["_src_prev"] = src_sig
st.session_state.setdefault("bars_memo", {})[interval] = st.session_state[
    "bars"
].copy()
st.session_state["bars_meta"] = {"symbol": symbol, "interval": interval}

pred = st.session_state["bars"]

# Normalise before anything reads them: the chart is built further down,
# and a note that arrived malformed -- from a hand-edited file, say --
# would otherwise take out the chart, the list and the save button.
st.session_state["notes"] = store.clean_notes(st.session_state.get("notes"))

if note := st.session_state.pop("_load_note", None):
    st.success(note, icon="✅")
if err := st.session_state.pop("_load_error", None):
    st.error(err)

# ------------------------------------------------------------ header strip

last_close = float(hist["Close"].iloc[-1])
prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
end_close = float(pred["Close"].iloc[-1])
move = (end_close / last_close - 1.0) * 100.0
day_move = (last_close / prev_close - 1.0) * 100.0 if prev_close else 0.0

head = st.columns([3, 2, 2, 2, 2])
head[0].markdown(f"### {symbol}")
head[0].caption(f"{interval} bars · {len(hist):,} loaded · {hist.index[-1]:%d %b %Y}")
head[1].metric("Last close", f"{last_close:,.2f}", f"{day_move:+.2f}%")
head[2].metric("Scenario end", f"{end_close:,.2f}", f"{move:+.2f}%")
head[3].metric("Bars ahead", f"{horizon}")
head[4].metric("Source", src_name.split(" - ")[-1].title())

# --------------------------------------------------- indicator selection

# Price style sits with the indicator pills rather than in the sidebar:
# it changes how the chart reads, so it belongs next to the chart.
pill_col, style_col = st.columns([5, 2], vertical_alignment="center")
chosen = pill_col.pills(
    "Indicators",
    INDICATORS,
    selection_mode="multi",
    default=["Moving averages", "RSI"],
    label_visibility="collapsed",
    key="chosen",
)
chosen = chosen or []
chart_style = style_col.segmented_control(
    "Price style",
    list(charting.STYLES),
    default=charting.CANDLES,
    label_visibility="collapsed",
    key="chart_style",
) or charting.CANDLES

# Fibonacci leg controls sit here, beside the chart they act on, rather than
# inside a collapsed sidebar expander where nobody finds them.
fib_anchor = "Auto (last impulse)"
fib_from = fib_to = 0
picking = False
if "Trend & Fibonacci" in chosen:
    lc, sc_, bc = st.columns([3, 4, 1.4], vertical_alignment="bottom")
    fib_anchor = lc.radio(
        "Fibonacci leg",
        ["Auto (last impulse)", "Choose exact points", "Drag a box"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if fib_anchor == "Drag a box":
        picking = True
        box = st.session_state.get("_fib_box")
        if box is None:
            sc_.info(
                "Drag a box on the chart from one end of the swing to the "
                "other — the box's height is the leg.",
                icon="🔲",
            )
        else:
            sc_.success(
                f"Leg {box['start_price']:,.2f} ({pd.Timestamp(box['start_ts']):%d %b})"
                f" → {box['end_price']:,.2f} ({pd.Timestamp(box['end_ts']):%d %b}). "
                f"Drag again to redraw.",
                icon="✅",
            )
        if bc.button("Clear", width="stretch"):
            st.session_state.pop("_fib_box", None)
            st.rerun()

if "Trend & Fibonacci" in chosen and fib_anchor == "Drag a box" and (
    st.session_state.get("_fib_box")
):
    # Fine-tune the box numerically. Everything -- the rectangle, the leg
    # line and every level -- is derived from these four numbers, so they
    # always move together and can never drift out of agreement. Dragging
    # the rectangle in the browser could not offer that: plotly does not
    # report the new position back, so the levels would stay behind.
    _b = st.session_state["_fib_box"]
    _dates = [d.strftime("%Y-%m-%d") for d in hist.index]

    def _idx_of(v, fallback):
        try:
            return _dates.index(pd.Timestamp(v).strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            return fallback

    e1, e2, e3, e4, e5 = st.columns([2, 2, 2, 2, 2])
    nb_from = e1.selectbox(
        "Box from", _dates, index=_idx_of(_b["start_ts"], 0), key="fbx_from"
    )
    nb_to = e2.selectbox(
        "to", _dates, index=_idx_of(_b["end_ts"], len(_dates) - 1), key="fbx_to"
    )
    step = max(round(float(_b["hi"] - _b["lo"]) / 100.0, 4), 0.01)
    nb_lo = e3.number_input("Bottom", value=float(_b["lo"]), step=step, key="fbx_lo")
    nb_hi = e4.number_input("Top", value=float(_b["hi"]), step=step, key="fbx_hi")
    shift = e5.number_input(
        "Shift both", value=0.0, step=step, key="fbx_shift",
        help="Moves the whole box up or down, keeping its height.",
    )

    lo_v, hi_v = sorted((float(nb_lo) + shift, float(nb_hi) + shift))
    rising_v = float(_b["end_price"]) >= float(_b["start_price"])
    updated = {
        "start_ts": str(pd.Timestamp(nb_from)),
        "end_ts": str(pd.Timestamp(nb_to)),
        "start_price": lo_v if rising_v else hi_v,
        "end_price": hi_v if rising_v else lo_v,
        "lo": lo_v,
        "hi": hi_v,
    }
    if updated != _b:
        st.session_state["_fib_box"] = updated
        st.rerun()
    elif fib_anchor == "Choose exact points":
        sc_.caption(
            "Pick the bar and the price for each end of the leg. "
            "Low → High measures a rally; High → Low measures a decline."
        )
    else:
        sc_.caption("Leg detected from the latest swing. Switch to a manual "
                    "mode to place it yourself.")

if "Trend & Fibonacci" in chosen and fib_anchor == "Choose exact points":
    # Deterministic alternative to clicking: name the two bars and which
    # price on each. Defaults land on the auto-detected swing, so this
    # starts from a sensible leg and gets nudged rather than built blind.
    auto = trends.last_impulse(hist, int(swing_n), int(swing_n))
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]

    def _pos(ts, fallback: int) -> int:
        try:
            return dates.index(pd.Timestamp(ts).strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            return fallback

    d1, f1, d2, f2 = st.columns(4)
    a_date = d1.selectbox(
        "Leg starts", dates,
        index=_pos(auto.start_ts if auto else None, max(0, len(dates) - 60)),
    )
    a_field = f1.selectbox("at its", ["Low", "High", "Close", "Open"], index=0)
    b_date = d2.selectbox(
        "Leg ends", dates,
        index=_pos(auto.end_ts if auto else None, len(dates) - 1),
    )
    b_field = f2.selectbox(
        "at its", ["High", "Low", "Close", "Open"], index=0, key="b_field"
    )

full = pd.concat([hist, pred])
overlays: dict[str, pd.Series] = {}
panels: list[tuple[str, dict[str, pd.Series]]] = []
active: dict[str, dict[str, pd.Series]] = {}

if "Parabolic SAR" in chosen:
    res = ind.parabolic_sar(full, af0=sar_af0, step=sar_step, af_max=sar_max)
    overlays["SAR"] = res["SAR"]
    active["Parabolic SAR"] = res

if "Moving averages" in chosen:
    ma_series: dict[str, pd.Series] = {}
    for _, row in ma_cfg.iterrows():
        if not bool(row["On"]):
            continue
        n, kind = int(row["Period"]), str(row["Type"])
        ma_series[f"{kind}{n}"] = ind.moving_average(full["Close"], n, kind)
    overlays |= ma_series
    if ma_series:
        active["Moving Average"] = ma_series

if "RSI" in chosen:
    s = ind.rsi(full, int(rsi_len))
    m = {s.name: s}
    panels.append(("RSI", m))
    active["RSI"] = m

if "MACD" in chosen:
    m = ind.macd(full, int(macd_fast), int(macd_slow), int(macd_sig))
    panels.append(("MACD", m))
    active["MACD"] = m

if "DMI / ADX" in chosen:
    m = ind.dmi(full, int(dmi_len), int(adx_len))
    panels.append(("DMI / ADX", m))
    active["DMI / ADX"] = m

if "Stochastic" in chosen:
    m = ind.stochastic(full, int(st_k), int(st_sk), int(st_d))
    panels.append(("Stochastic", m))
    active["Stochastic"] = m

if "Price Oscillator" in chosen:
    m = ind.price_oscillator(
        full, int(po_fast), int(po_slow), int(po_sig), po_kind, po_pct
    )
    panels.append(("Price Oscillator", m))
    active["Price Oscillator"] = m

if "QQE" in chosen:
    m = ind.qqe(full, int(qqe_rsi), int(qqe_sf), float(qqe_f))
    panels.append(("QQE", m))
    active["QQE"] = m

# ------------------------------------------------- trend and Fibonacci

fib_levels: dict[str, float] = {}
overlay_shapes: dict = {}

if "Trend & Fibonacci" in chosen:
    # Clicks arrive from the previous run's chart, so they are read from
    # session state before this run's figure is built.
    if fib_anchor == "Drag a box":
        before = st.session_state.get("_fib_box")
        sel = (st.session_state.get("chart") or {}).get("selection") or {}
        boxes = sel.get("box") or []
        if boxes:
            try:
                b = boxes[0]
                xs = sorted(pd.Timestamp(v) for v in b["x"])
                ys = sorted(float(v) for v in b["y"])
                x0, x1 = xs[0], xs[-1]
                lo, hi = ys[0], ys[-1]

                # A rectangle has no direction of its own. Read it from the
                # price action inside: a rally is measured low-to-high, a
                # decline high-to-low, which is how the tool is drawn.
                inside = hist.loc[(hist.index >= x0) & (hist.index <= x1)]
                rising = (
                    float(inside["Close"].iloc[-1]) >= float(inside["Close"].iloc[0])
                    if len(inside) > 1
                    else True
                )
                st.session_state["_fib_box"] = {
                    "start_ts": str(x0),
                    "end_ts": str(x1),
                    "start_price": lo if rising else hi,
                    "end_price": hi if rising else lo,
                    "lo": lo,
                    "hi": hi,
                }
            except Exception:  # noqa: BLE001 - ignore an unreadable drag
                pass
        # The control row has already drawn by the time the drag is read, so
        # its status line would lag a run behind. Guarded on a real change.
        if st.session_state.get("_fib_box") != before:
            st.rerun()

    # Detected on history alone: the levels are a reference your scenario is
    # measured against, so they must not move as you redraw the bars.
    if fib_anchor == "Auto (last impulse)":
        imp = trends.last_impulse(hist, int(swing_n), int(swing_n))
    elif fib_anchor == "Drag a box":
        box = st.session_state.get("_fib_box")
        imp = (
            trends.Impulse(
                pd.Timestamp(box["start_ts"]), float(box["start_price"]),
                pd.Timestamp(box["end_ts"]), float(box["end_price"]),
            )
            if box
            else None
        )
    else:
        ts_a, ts_b = pd.Timestamp(a_date), pd.Timestamp(b_date)
        imp = (
            trends.Impulse(
                ts_a, float(hist.loc[ts_a, a_field]),
                ts_b, float(hist.loc[ts_b, b_field]),
            )
            if ts_a != ts_b
            else None
        )

    if imp is None:
        if fib_anchor == "Choose exact points":
            st.warning("The leg needs two different bars.")
        elif fib_anchor == "Drag a box":
            pass  # the control row already says what to do
        elif fib_anchor == "Auto (last impulse)":
            st.info(
                "No swing found at this sensitivity — lower it, or place the "
                "leg yourself with **Choose exact points**."
            )
    else:
        if fib_anchor == "Choose exact points":
            st.caption(
                f"Leg: {imp.start_price:,.2f} ({a_field}, {imp.start_ts:%d %b}) → "
                f"{imp.end_price:,.2f} ({b_field}, {imp.end_ts:%d %b})  ·  "
                f"{imp.pct:+.2f}%"
            )
        fib_levels = trends.fib_levels(
            imp,
            retracements=tuple(float(r) for r in fib_ratios),
            projections=tuple(float(r) for r in fib_projs),
        )
        overlay_shapes = {
            "impulse": imp,
            "levels": fib_levels,
            # Extend across the scenario so you can see what your bars hit.
            "span": (imp.start_ts, full.index[-1]),
            "trend_lines": {},
        }
        if show_trendline:
            anchors = trends.trend_line(hist, int(swing_n), int(swing_n))
            if anchors is not None:
                overlay_shapes["trend_lines"] = {
                    "trend": trends.project(anchors, full.index)
                }

# ------------------------------------------------------------------ chart

# Computed before the chart so the same events drive both the on-chart
# markers and the table underneath.
events = (
    signals.collect(active, full["Close"], pred.index, fib=fib_levels)
    if (active or fib_levels)
    else pd.DataFrame()
)

view_hist = hist.tail(int(show_tail))
view_from = view_hist.index[0]
overlays_v = {k: v.loc[v.index >= view_from] for k, v in overlays.items()}

# Trend lines are clipped to the view like any other overlay; the Fibonacci
# grid keeps its own span so the levels stay anchored to the leg even when
# the leg itself has scrolled off the left edge.
shapes_v = dict(overlay_shapes)
if shapes_v.get("trend_lines"):
    shapes_v["trend_lines"] = {
        k: v.loc[v.index >= view_from] for k, v in shapes_v["trend_lines"].items()
    }
if shapes_v.get("span"):
    shapes_v["span"] = (max(shapes_v["span"][0], view_from), shapes_v["span"][1])
panels_v = [
    (n, {k: v.loc[v.index >= view_from] for k, v in m.items() if k != "trend"})
    for n, m in panels
]

# .streamlit/config.toml pins the app to dark; honour an override if the
# viewer switches themes, and fall back to dark when unavailable.
try:
    is_dark = st.context.theme.type != "light"
except Exception:  # noqa: BLE001 - theme unavailable in bare/test runs
    is_dark = True

fig = charting.build(
    view_hist,
    pred,
    overlays_v,
    panels_v,
    symbol=symbol,
    log_scale=log_scale,
    dark=is_dark,
    crosshair=crosshair,
    unified_hover=unified_hover,
    events=events,
    overlay_shapes=shapes_v,
    pickable=(
        {
            f: view_hist[f]
            for f in st.session_state.get("_snap_fields", ["High", "Low"])
        }
        if picking
        else None
    ),
    dragmode="select" if picking else "pan",
    style=chart_style,
    notes=st.session_state.get("notes", []),
)
st.plotly_chart(
    fig,
    width="stretch",
    key="chart",
    on_select="rerun",
    selection_mode=["points", "box"],
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
        # Freehand annotation. Pick a tool, drag on the chart, and use the
        # eraser to remove one. These are drawn by you and are not read
        # back into any calculation.
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawrect",
            "eraseshape",
        ],
        "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"],
        # Notes and freehand shapes can be dragged around in the browser.
        # Plotly does not report the new position back, so a drag is a quick
        # visual nudge; the Nudge columns under the table make it permanent.
        "edits": {
            "annotationPosition": True,
            "annotationTail": True,
            "shapePosition": True,
        },
        # Skip the hover-driven redraw plotly does while a drag is in flight.
        "plotGlPixelRatio": 1,
    },
)

# ----------------------------------------------------- bars + consequences

st.divider()
edit_col, sig_col = st.columns([5, 6], gap="large")

with edit_col:
    st.markdown("#### Your bars")
    st.caption("Edit any cell — the chart updates. High/Low bracket the body.")
    edited = st.data_editor(
        st.session_state["bars"],
        width="stretch",
        num_rows="fixed",
        height=min(38 * (horizon + 1) + 8, 460),
        column_config={
            "Open": st.column_config.NumberColumn(format="%.2f"),
            "High": st.column_config.NumberColumn(format="%.2f"),
            "Low": st.column_config.NumberColumn(format="%.2f"),
            "Close": st.column_config.NumberColumn(format="%.2f"),
            "Volume": st.column_config.NumberColumn(format="%.0f"),
        },
        # Rebuild the editor whenever the bar set is replaced wholesale,
        # otherwise it keeps showing the previous rows.
        key=f"editor_{symbol}_{interval}_{len(pred)}_{src_name}",
    )
    clean = predictors.sanitize(edited)
    if not clean.equals(st.session_state["bars"]):
        st.session_state["bars"] = clean
        st.rerun()
    pred = clean

    # Two buttons, always visible. The file goes to whoever's device is
    # using the app, so it is theirs and survives any redeploy.
    scen_name = st.session_state.get("scen_name", "").strip()
    label = scen_name or f"{symbol}-{interval}"

    save_col, open_col = st.columns(2)
    save_col.download_button(
        "💾  Save bars",
        data=store.to_json_bytes(
            label, symbol, interval, pred, st.session_state.get("notes", [])
        ),
        file_name=store.filename_for(label, symbol, interval),
        mime="application/json",
        width="stretch",
        type="primary",
        help=f"Downloads “{store.filename_for(label, symbol, interval)}” to "
        f"your device. Open it again any time, on any computer.",
    )
    open_it = open_col.button(
        "📂  Open bars",
        width="stretch",
        help="Load a file you saved earlier. It is converted to whatever "
        "interval you are looking at.",
    )
    if open_it:
        st.session_state["_show_opener"] = True

    st.text_input(
        "Name it (optional)", "", placeholder=label, key="scen_name",
        label_visibility="collapsed",
    )

    if st.session_state.get("_show_opener"):
        up = st.file_uploader(
            "Choose a saved .json file", type=["json"], key="scen_upload"
        )
        if up is not None:
            # The uploader keeps returning the same file on every rerun;
            # only act when a genuinely different one arrives.
            stamp = (up.name, up.size)
            if st.session_state.get("_last_upload") != stamp:
                st.session_state["_last_upload"] = stamp
                try:
                    sc = store.from_json_bytes(up.getvalue(), Path(up.name).stem)
                    st.session_state["_pending_scenario"] = sc
                    st.session_state["_show_opener"] = False
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read that file: {e}")

    # Server-side files are only offered to someone sitting at the machine
    # running the app -- see _viewer_is_local.
    if _show_local_files():
        with st.expander("Keep a copy on this computer"):
            st.caption(f"Stored in `{store.DIR.name}/` next to the app.")
            if st.button(
                "Save here", width="stretch", disabled=not scen_name
            ):
                try:
                    p = store.save(
                        scen_name, symbol, interval, pred,
                        st.session_state.get("notes", []),
                    )
                    st.success(f"Saved → {p.name}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not save: {e}")

            saved = store.list_all()
            if saved:
                pick = st.selectbox("Saved here", ["—"] + [s.label for s in saved])
                hit = next((s for s in saved if s.label == pick), None)
                if hit is not None:
                    if hit.interval != interval:
                        st.caption(
                            resample.describe(hit.bars, hit.interval, interval)
                        )
                    lc, dc = st.columns(2)
                    if lc.button("Open", width="stretch", type="primary"):
                        st.session_state["_pending_scenario"] = hit
                        st.rerun()
                    if dc.button("Delete", width="stretch"):
                        store.delete(hit.name)
                        st.rerun()

with sig_col:
    with st.expander(f"Notes on the chart ({len(st.session_state['notes'])})"):
        nt, nd, npr, nb = st.columns([5, 3, 2, 1.6], vertical_alignment="bottom")
        note_text = nt.text_input(
            "Note", "", placeholder="e.g. gap fills here", key="note_text",
            label_visibility="collapsed",
        )
        # Any date, not just one with a bar on it: a note may want to sit in
        # a gap, past the last bar, or anywhere else that reads well.
        note_date = nd.date_input(
            "On",
            value=full.index[-1].date(),
            label_visibility="collapsed",
            key="note_date",
        )
        note_price = npr.number_input(
            "At",
            value=round(float(full["Close"].iloc[-1]), 2),
            step=1.0,
            label_visibility="collapsed",
            key="note_price",
        )
        if nb.button("Add", width="stretch", disabled=not note_text.strip()):
            st.session_state.setdefault("notes", []).append(
                {
                    "ts": str(pd.Timestamp(note_date)),
                    "price": float(note_price),
                    "text": note_text.strip(),
                }
            )
            st.rerun()

        if st.session_state["notes"]:
            st.caption(
                "Drag a note on the chart to move it. To make a position "
                "stick, set the nudge here — a browser drag is not readable "
                "back, so it resets on the next redraw."
            )
            table = pd.DataFrame(st.session_state["notes"])
            table["ts"] = pd.to_datetime(table["ts"]).dt.strftime("%Y-%m-%d")
            edited_notes = st.data_editor(
                table[["text", "ts", "price", "dx", "dy", ]],
                width="stretch",
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "text": st.column_config.TextColumn("Note", width="large"),
                    "ts": st.column_config.TextColumn("Points at", width="small"),
                    "price": st.column_config.NumberColumn(
                        "Price", format="%.2f", width="small"
                    ),
                    "dx": st.column_config.NumberColumn(
                        "Nudge →", help="Pixels right of the point", step=5
                    ),
                    "dy": st.column_config.NumberColumn(
                        "Nudge ↑", help="Pixels above the point (negative)", step=5
                    ),
                },
                key="notes_editor",
            )
            cleaned_notes = store.clean_notes(edited_notes.to_dict("records"))
            if cleaned_notes != st.session_state["notes"]:
                st.session_state["notes"] = cleaned_notes
                st.rerun()

    st.markdown("#### Triggered signals")
    if not active:
        st.caption("Pick indicators above the chart to detect events.")
    else:
        ev = events
        if ev.empty:
            st.caption("Nothing fires inside your predicted bars.")
            st.success("No indicator events triggered by this scenario.")
        else:
            bull = int((ev["Bias"] == "bullish").sum())
            bear = int((ev["Bias"] == "bearish").sum())
            st.caption(
                f"{len(ev)} event(s) — {bull} bullish, {bear} bearish. "
                "Marked on the chart with triangles."
            )
            st.dataframe(
                ev.drop(columns=["_ts"]),
                width="stretch",
                hide_index=True,
                height=min(38 * (len(ev) + 1) + 8, 460),
                column_config={
                    "Bar": st.column_config.TextColumn(width="small"),
                    "Indicator": st.column_config.TextColumn(width="small"),
                    "Event": st.column_config.TextColumn(width="medium"),
                    "Bias": st.column_config.TextColumn(width="small"),
                },
            )

# ------------------------------------------------------------------ detail

with st.expander("Indicator detail — before/after and raw values"):
    if not active:
        st.caption("No indicators selected.")
    else:
        t1, t2 = st.tabs(["Before / after", "Values"])
        with t1:
            st.dataframe(
                signals.snapshot(active, hist.index[-1], pred.index[-1]),
                width="stretch",
                hide_index=True,
            )
        with t2:
            frame = pd.DataFrame(index=full.index)
            for name, m in active.items():
                for label, s in m.items():
                    if label != "trend":
                        frame[f"{name} · {label}"] = s
            st.dataframe(frame.tail(int(horizon) + 15).round(3), width="stretch")
            st.download_button(
                "Download full indicator table (CSV)",
                frame.to_csv().encode(),
                file_name=f"{symbol}_indicators.csv",
                mime="text/csv",
            )
