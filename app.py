"""BIST bar-scenario dashboard.

Pick a symbol, propose the next few bars (draw them yourself or let a source
generate them), then switch indicators on and off to see what your scenario
does to them.
"""

from __future__ import annotations

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

INDICATORS = [
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


def _seed_key(symbol: str, interval: str, horizon: int, last: pd.Timestamp) -> tuple:
    return (symbol, interval, horizon, str(last))


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

    # A queued load has to be resolved here, before the horizon slider is
    # created: Streamlit refuses to let a widget's state be written once the
    # widget exists, and loading a scenario changes the bar count.
    if st.session_state.get("_pending_load"):
        try:
            sc = store.load(st.session_state.pop("_pending_load"))
            conv = resample.convert(sc.bars, sc.interval, interval)
            st.session_state["_loaded_bars"] = conv
            st.session_state["horizon"] = int(min(max(len(conv), 1), 60))
            st.session_state["_load_note"] = (
                f"Loaded “{sc.name}” ({sc.symbol} {sc.interval})"
                + (
                    f" → converted to {len(conv)} × {interval} bars"
                    if sc.interval != interval
                    else ""
                )
            )
        except Exception as e:  # noqa: BLE001
            st.session_state["_load_error"] = str(e)

    st.divider()
    st.subheader("Bars to predict")

    horizon = st.slider("How many bars", 1, 60, 5, key="horizon")
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
    st.subheader("Saved scenarios")

    saved = store.list_all()
    if saved:
        pick = st.selectbox(
            "Load a scenario",
            ["—"] + [sc.label for sc in saved],
            help="Scenarios saved at another interval are converted to the "
            "current one on load.",
        )
        chosen_sc = next((sc for sc in saved if sc.label == pick), None)
        if chosen_sc is not None:
            if chosen_sc.interval != interval:
                st.caption(
                    resample.describe(chosen_sc.bars, chosen_sc.interval, interval)
                )
            lc, dc = st.columns(2)
            if lc.button("Load", width="stretch", type="primary"):
                st.session_state["_pending_load"] = chosen_sc.name
                st.rerun()
            if dc.button("Delete", width="stretch"):
                store.delete(chosen_sc.name)
                st.rerun()
    else:
        st.caption("None saved yet.")

    save_name = st.text_input("Save current bars as", "", placeholder="e.g. breakout")
    do_save = st.button("Save", width="stretch", disabled=not save_name.strip())

    st.divider()
    st.subheader("Indicator settings")
    st.caption("Switch indicators on above the chart; tune them here.")

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

key = _seed_key(symbol, interval, horizon, hist.index[-1])

loaded = st.session_state.pop("_loaded_bars", None)
if loaded is not None:
    # Re-anchor onto the current future index; the saved timestamps belong
    # to whenever the scenario was drawn.
    n = min(len(loaded), len(fut_idx))
    frame = loaded.iloc[:n].copy()
    frame.index = fut_idx[:n]
    frame.index.name = "Date"
    st.session_state["bars"] = predictors.sanitize(frame)
    st.session_state["_key"] = key
elif regen or st.session_state.get("_key") != key or "bars" not in st.session_state:
    st.session_state["_key"] = key
    st.session_state["bars"] = predictors.build(src_name, **cfg).propose(hist, fut_idx)
elif st.session_state.get("_src") != (src_name, tuple(sorted(cfg.items()))):
    st.session_state["bars"] = predictors.build(src_name, **cfg).propose(hist, fut_idx)
st.session_state["_src"] = (src_name, tuple(sorted(cfg.items())))

pred = st.session_state["bars"]

if note := st.session_state.pop("_load_note", None):
    st.success(note, icon="✅")
if err := st.session_state.pop("_load_error", None):
    st.error(f"Could not load scenario: {err}")

if do_save:
    try:
        p = store.save(save_name, symbol, interval, pred)
        st.success(f"Saved “{save_name}” → {p.name}", icon="💾")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not save: {e}")

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

chosen = st.pills(
    "Indicators",
    INDICATORS,
    selection_mode="multi",
    default=["Moving averages", "RSI"],
    label_visibility="collapsed",
    key="chosen",
)
chosen = chosen or []

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

# ------------------------------------------------------------------ chart

# Computed before the chart so the same events drive both the on-chart
# markers and the table underneath.
events = (
    signals.collect(active, full["Close"], pred.index)
    if active
    else pd.DataFrame()
)

view_hist = hist.tail(int(show_tail))
view_from = view_hist.index[0]
overlays_v = {k: v.loc[v.index >= view_from] for k, v in overlays.items()}
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
)
st.plotly_chart(
    fig,
    width="stretch",
    key="chart",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
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
        key=f"editor_{key}_{src_name}",
    )
    clean = predictors.sanitize(edited)
    if not clean.equals(st.session_state["bars"]):
        st.session_state["bars"] = clean
        st.rerun()

with sig_col:
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
