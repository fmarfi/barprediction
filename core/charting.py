"""Plotly figure assembly: price panel plus one subplot per oscillator."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP = "#26a69a"
DOWN = "#ef5350"
PRED_UP = "#66bb6a"
PRED_DOWN = "#ff7043"

MA_COLORS = ["#2f7fd1", "#9c37b8", "#e0821a", "#0f9aa8", "#8d6e63", "#607d8b"]
SAR_COLOR = "#e5a52b"

# The page background is left transparent so the figure sits on Streamlit's
# own surface; everything else has to be stated per theme or it inherits
# plotly_dark's light-on-light text when the app is in light mode.
THEMES = {
    "dark": {
        "template": "plotly_dark",
        "text": "#c8d0de",
        "muted": "#93a0b5",
        "grid": "rgba(130,140,160,0.16)",
        "hover_bg": "rgba(22,26,34,0.94)",
        "spike": "rgba(160,170,190,0.55)",
        "shade": "rgba(120,145,190,0.12)",
        "divider": "rgba(150,150,150,0.65)",
        "level": "rgba(150,155,165,0.45)",
        "tag_bg": "rgba(14,17,23,0.92)",
        "watermark": "rgba(150,165,190,0.045)",
    },
    "light": {
        "template": "plotly_white",
        "text": "#2f3945",
        "muted": "#5d6b7d",
        "grid": "rgba(110,120,140,0.18)",
        "hover_bg": "rgba(255,255,255,0.96)",
        "spike": "rgba(90,100,120,0.55)",
        "shade": "rgba(70,105,175,0.09)",
        "divider": "rgba(90,95,105,0.6)",
        "level": "rgba(110,118,130,0.5)",
        "tag_bg": "rgba(255,255,255,0.92)",
        "watermark": "rgba(60,75,100,0.05)",
    },
}


CANDLES = "Candles"
BARS = "OHLC bars"
LINE_STYLE = "Line"
STYLES = (CANDLES, BARS, LINE_STYLE)


def _price_trace(
    df: pd.DataFrame,
    name: str,
    style: str,
    up_line: str,
    down_line: str,
    up_fill: str,
    down_fill: str,
):
    """Price series as candles, OHLC bars, or a plain close line."""
    if style == LINE_STYLE:
        return go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=name,
            line=dict(width=1.6, color=up_line),
            hovertemplate="%{y:,.2f}<extra>" + name + "</extra>",
        )

    common = dict(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=name,
        increasing_line_color=up_line,
        decreasing_line_color=down_line,
        line_width=1,
    )
    if style == BARS:
        # OHLC bars are drawn from lines only, so they take no fill.
        return go.Ohlc(**common)
    return go.Candlestick(
        **common, increasing_fillcolor=up_fill, decreasing_fillcolor=down_fill
    )


def _dt(ts):
    """Plain datetime for layout shapes.

    Traces convert pandas Timestamps themselves, but shapes and annotations
    keep whatever object they are handed. Streamlit's JSON encoder copes;
    kaleido's static export does not, so normalise at the boundary.
    """
    return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts


def build(
    history: pd.DataFrame,
    predicted: pd.DataFrame,
    overlays: dict[str, pd.Series],
    panels: list[tuple[str, dict[str, pd.Series]]],
    *,
    symbol: str,
    log_scale: bool = False,
    dark: bool = True,
    crosshair: bool = False,
    unified_hover: bool = False,
    events: pd.DataFrame | None = None,
    overlay_shapes: dict | None = None,
    pickable: dict[str, pd.Series] | None = None,
    dragmode: str = "pan",
    style: str = CANDLES,
    notes: list[dict] | None = None,
    interval: str = "",
    right_pad: float = 0.09,
    height_price: int = 520,
    height_panel: int = 165,
) -> go.Figure:
    """Compose the chart.

    `overlays` are drawn on the price axis; `panels` each get their own row.
    History and predicted bars share one continuous x-axis so indicators run
    straight through the boundary.

    `crosshair` and `unified_hover` are off by default: spikes drawn across
    every subplot and a hover box that gathers all traces both re-render the
    whole figure on each mouse move, which is what makes panning feel heavy.
    """
    t = THEMES["dark" if dark else "light"]

    rows = 1 + len(panels)
    heights = [height_price] + [height_panel] * len(panels)
    total = sum(heights)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[h / total for h in heights],
        subplot_titles=[""] + [name for name, _ in panels],
    )

    fig.add_trace(
        _price_trace(history, "History", style, UP, DOWN, UP, DOWN),
        row=1,
        col=1,
    )

    if not predicted.empty:
        fig.add_trace(
            _price_trace(
                predicted,
                "Your bars",
                style,
                PRED_UP,
                PRED_DOWN,
                "rgba(102,187,106,0.45)",
                "rgba(255,112,67,0.45)",
            ),
            row=1,
            col=1,
        )
        # Shade the scenario so predicted bars never read as real history.
        fig.add_vrect(
            x0=_dt(history.index[-1]),
            x1=_dt(predicted.index[-1]),
            fillcolor=t["shade"],
            line_width=0,
            layer="below",
        )
        fig.add_vline(
            x=_dt(history.index[-1]),
            line_width=1,
            line_dash="dot",
            line_color=t["divider"],
        )
        fig.add_annotation(
            x=_dt(predicted.index[-1]),
            y=1,
            yref="y domain",
            text="scenario",
            showarrow=False,
            font=dict(size=10, color=t["muted"]),
            xanchor="right",
            yanchor="bottom",
            row=1,
            col=1,
        )

    if pickable:
        # Candlesticks do not report clicks usefully, so markers sit on top
        # to catch them; Streamlit reads the selection back and turns it
        # into Fibonacci anchors. Deliberately visible -- a transparent
        # marker is neither reliably clickable nor discoverable.
        #
        # One trace per price field, so the anchor lands on the exact value
        # meant. A Fibonacci leg normally runs low-to-high, and snapping
        # everything to the close would quietly measure the wrong swing.
        for field, series in pickable.items():
            if series is None or series.empty:
                continue
            # Not `style`: that name holds the chart style parameter, and
            # rebinding it here would quietly break anything below.
            pick = PICK_STYLES.get(field, PICK_STYLES["Close"])
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="markers",
                    name=field,
                    marker=dict(
                        size=pick["size"],
                        symbol=pick["symbol"],
                        color=pick["color"],
                        line=dict(width=1, color=pick["edge"]),
                    ),
                    hovertemplate=(
                        "%{x|%d %b %Y}<br>" + field + " %{y:,.2f}"
                        "<extra>click to anchor</extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Status line: the last bar's OHLC, as trading platforms print above the
    # chart. Static rather than following the cursor, because hover stays in
    # the browser and never reaches the server.
    last = (predicted if not predicted.empty else history).iloc[-1]
    prev_close = float(history["Close"].iloc[-1] if not predicted.empty else (
        history["Close"].iloc[-2] if len(history) > 1 else last["Close"]
    ))
    chg = float(last["Close"]) - prev_close
    pct = chg / prev_close * 100.0 if prev_close else 0.0
    tone = UP if chg >= 0 else DOWN
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        yshift=34,
        showarrow=False,
        align="left",
        text=(
            f"<b>{symbol}</b>  <span style='color:{t['muted']}'>{interval}</span>   "
            f"<span style='color:{t['muted']}'>O</span> {last['Open']:,.2f}  "
            f"<span style='color:{t['muted']}'>H</span> {last['High']:,.2f}  "
            f"<span style='color:{t['muted']}'>L</span> {last['Low']:,.2f}  "
            f"<span style='color:{t['muted']}'>C</span> {last['Close']:,.2f}  "
            f"<span style='color:{tone}'>{chg:+,.2f} ({pct:+.2f}%)</span>"
        ),
        font=dict(size=11, color=t["text"]),
    )

    _add_overlays(fig, overlays)

    if overlay_shapes:
        _add_trend_and_fib(fig, overlay_shapes, t)

    _add_notes(fig, notes, t)

    if events is not None and not events.empty and not predicted.empty:
        _add_event_markers(fig, events, predicted, t)

    # Tag the closing level at the right edge, matching where the panel tags
    # sit. With a scenario present that is the scenario's end, not the last
    # real close -- anchoring it to history would drop the label on top of
    # the predicted candles.
    edge_close = predicted["Close"] if not predicted.empty else history["Close"]
    _tag_last(fig, edge_close, "#c8d0de" if dark else "#2f3945", row=1, t=t)

    for i, (name, series_map) in enumerate(panels, start=2):
        _add_panel(fig, name, series_map, row=i, t=t)

    fig.update_layout(
        height=total,
        # Right margin holds the price axis ticks and, beyond them, the
        # last-value badges.
        margin=dict(l=4, r=96, t=74, b=4),
        xaxis_rangeslider_visible=False,
        hovermode="x unified" if unified_hover else "x",
        hoverdistance=8,
        # Keep pan/zoom across Streamlit reruns: editing a bar re-sends the
        # figure, and without this the view snaps back to full extent.
        uirevision=f"{symbol}|{len(panels)}",
        transition_duration=0,
        hoverlabel=dict(
            bgcolor=t["hover_bg"], font_size=11, font_color=t["text"]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=t["text"]),
            itemsizing="constant",
        ),
        template=t["template"],
        font=dict(color=t["text"], size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # "pan" for normal use; "select" while picking anchors, because a
        # pan drag swallows the gesture and Streamlit never sees a
        # selection -- clicks simply do nothing.
        dragmode=dragmode,
        bargap=0.15,
        # Style for anything drawn with the modebar tools.
        newshape=dict(
            line=dict(color="#5fa8d3", width=1.6),
            fillcolor="rgba(95,168,211,0.10)",
            opacity=0.9,
            layer="above",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=t["grid"],
        rangeslider_visible=False,
        tickfont=dict(color=t["muted"], size=10),
    )
    if crosshair:
        # spikesnap="data" is markedly cheaper than "cursor", which recomputes
        # on every pixel of mouse movement.
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="data",
            spikethickness=1,
            spikedash="dot",
            spikecolor=t["spike"],
        )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid"],
        zeroline=False,
        tickfont=dict(color=t["muted"], size=10),
        # Prices on the right, trading-platform style.
        side="right",
        # fixedrange=False is what makes the y axis draggable/scrollable on
        # its own: drag the axis to stretch it, double-click to autoscale.
        fixedrange=False,
    )
    fig.update_xaxes(fixedrange=False)
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    # Panel titles: small and left-aligned, not centred headings.
    titles = {name for name, _ in panels}
    for ann in fig.layout.annotations:
        if ann.text in titles:
            ann.update(
                x=0, xanchor="left", font=dict(size=11, color=t["muted"])
            )

    # Hide weekend gaps so daily candles sit flush.
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # Breathing room past the last bar, the way trading platforms leave it.
    # Without it the newest bar is jammed against the price scale, and there
    # is nowhere to project a level or place a note ahead of price.
    spine = predicted if not predicted.empty else history
    if right_pad > 0 and len(spine) > 1 and len(history) > 1:
        step = pd.Series(history.index).diff().median()
        span = len(history) + len(predicted)
        if pd.notna(step) and step > pd.Timedelta(0):
            fig.update_xaxes(
                range=[
                    _dt(history.index[0]),
                    _dt(spine.index[-1] + step * max(1, round(span * right_pad))),
                ]
            )

    # Symbol and interval behind the candles, as a quiet reminder of what is
    # on screen -- the chart is often read without the surrounding page.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.55,
        text=f"{symbol}  {interval}".strip(),
        showarrow=False,
        font=dict(size=40, color=t["watermark"]),
        opacity=1.0,
    )

    # Plotly greys out everything outside a selection, and an empty
    # selection -- which is what a double-click leaves behind -- greys out
    # the entire chart. Anchors are drawn explicitly, so plotly's own
    # selection styling is never wanted here.
    fig.update_traces(selectedpoints=None)

    # Lock everything drawn from data: RSI's 50 line, the Fibonacci levels,
    # the scenario shading. Shape dragging stays on in the config so
    # freehand annotations can still be moved, but a reference level is not
    # something to nudge by accident -- and a dragged one would silently
    # disagree with the value it is named after.
    fig.update_shapes(editable=False)
    return fig


def _add_overlays(fig: go.Figure, overlays: dict[str, pd.Series]) -> None:
    ma_i = 0
    for name, s in overlays.items():
        if name.startswith("SAR"):
            fig.add_trace(
                # One marker per bar, so this is the heaviest overlay.
                # Skipping hover keeps it off the pan/zoom critical path.
                # (SVG, not Scattergl: at these point counts WebGL gains
                # nothing and disappears wherever WebGL is unavailable.)
                go.Scatter(
                    x=s.index,
                    y=s,
                    mode="markers",
                    name=name,
                    marker=dict(size=2.5, color=SAR_COLOR),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s,
                    mode="lines",
                    name=name,
                    line=dict(width=1.4, color=MA_COLORS[ma_i % len(MA_COLORS)]),
                    hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
                ),
                row=1,
                col=1,
            )
            ma_i += 1


def _tag_last(
    fig: go.Figure, s: pd.Series, color: str, row: int, t: dict, digits: int = 2
) -> None:
    """Badge a series' final value on the right-hand axis.

    Pinned to paper x=1 rather than to the last data point, so the badge
    stays glued to the axis when the chart is panned or zoomed.
    """
    sv = s.dropna()
    if sv.empty:
        return
    v = float(sv.iloc[-1])
    # Index levels run to five figures; two decimals there is noise and makes
    # the badge wide enough to crowd its neighbours.
    if abs(v) >= 1000:
        digits = 0
    fig.add_annotation(
        xref="paper",
        x=1.0,
        xanchor="left",
        # Clear the tick labels rather than overlapping them; the right
        # margin is sized to match.
        xshift=36,
        yref="y" if row == 1 else f"y{row}",
        y=v,
        text=f"{v:,.{digits}f}",
        showarrow=False,
        font=dict(size=9, color=color),
        bgcolor=t["tag_bg"],
        borderpad=2,
        opacity=0.95,
    )


FIB_COLORS = {
    "0": "#8e99ab",
    "0.236": "#4aa3df",
    "0.382": "#37a86b",
    "0.5": "#e0a020",
    "0.618": "#e05c3a",
    "0.786": "#9b5cc9",
    "1": "#8e99ab",
}
FIB_PROJECTION = "#7e8aa0"

# Anchor dots, shaped so the field is obvious at a glance: highs point up
# and sit on the wick tops, lows point down.
PICK_STYLES = {
    "High": {
        "symbol": "triangle-up", "size": 7,
        "color": "rgba(38,166,154,0.55)", "edge": "rgba(90,220,205,0.95)",
    },
    "Low": {
        "symbol": "triangle-down", "size": 7,
        "color": "rgba(239,83,80,0.55)", "edge": "rgba(255,140,135,0.95)",
    },
    "Close": {
        "symbol": "circle", "size": 6,
        "color": "rgba(215,225,240,0.35)", "edge": "rgba(120,170,220,0.9)",
    },
    "Open": {
        "symbol": "diamond", "size": 6,
        "color": "rgba(200,190,240,0.35)", "edge": "rgba(170,150,230,0.9)",
    },
}


def _add_trend_and_fib(fig: go.Figure, shapes: dict, t: dict) -> None:
    """Draw the impulse leg, its trend line, and its Fibonacci grid."""
    imp = shapes.get("impulse")

    for name, line in (shapes.get("trend_lines") or {}).items():
        fig.add_trace(
            go.Scatter(
                x=line.index,
                y=line,
                mode="lines",
                name=name,
                line=dict(width=1.2, color="#5fa8d3", dash="longdash"),
                hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )

    if imp is not None:
        # The box around the leg -- the shape a Fibonacci tool is normally
        # dragged out as, so what the levels measure is unmistakable.
        fig.add_shape(
            type="rect",
            x0=_dt(imp.start_ts),
            x1=_dt(imp.end_ts),
            y0=min(imp.start_price, imp.end_price),
            y1=max(imp.start_price, imp.end_price),
            line=dict(color="rgba(150,170,200,0.55)", width=1, dash="dot"),
            fillcolor="rgba(150,170,200,0.06)",
            layer="below",
            row=1,
            col=1,
        )

        # The leg itself, so it is obvious what the levels are measured from.
        fig.add_trace(
            go.Scatter(
                x=[_dt(imp.start_ts), _dt(imp.end_ts)],
                y=[imp.start_price, imp.end_price],
                mode="lines+markers",
                name=f"impulse {imp.pct:+.1f}%",
                line=dict(width=1.8, color="#d8dee9"),
                marker=dict(size=7, symbol="circle-open", line=dict(width=1.6)),
                hovertemplate="%{y:.2f}<extra>impulse</extra>",
            ),
            row=1,
            col=1,
        )

    levels = shapes.get("levels") or {}
    if not levels:
        return

    x0 = _dt(shapes["span"][0])
    x1 = _dt(shapes["span"][1])
    for label, price in levels.items():
        projection = float(label) > 1.0
        color = FIB_PROJECTION if projection else FIB_COLORS.get(label, "#8e99ab")
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=price,
            y1=price,
            line=dict(
                color=color,
                width=1.6 if label in ("0.382", "0.5", "0.618") else 1,
                dash="dot" if projection else "solid",
            ),
            opacity=0.75,
            layer="below",
            row=1,
            col=1,
        )
        # Labels go on the left edge of the plot, not at the start of the
        # level. A short leg sits far to the right, where the labels would
        # pile onto each other and onto the candles.
        fig.add_annotation(
            xref="paper",
            x=0.0,
            xanchor="left",
            xshift=3,
            yref="y",
            y=price,
            text=f"{label}   {price:,.2f}",
            showarrow=False,
            yanchor="middle",
            font=dict(size=9, color=color),
            bgcolor=t["tag_bg"],
            borderpad=1,
            opacity=0.92,
        )


def _add_notes(fig: go.Figure, notes: list[dict] | None, t: dict) -> None:
    """Text boxes the user placed on the price panel."""
    for n in notes or []:
        try:
            text = str(n["text"]).strip()
            x = _dt(pd.Timestamp(n["ts"]))
            y = float(n["price"])
            # Offset of the box from what it points at, in pixels. Dragging
            # a note in the browser moves exactly this, and the same numbers
            # are editable in the table so a position can be made permanent.
            dx = float(n.get("dx", 0) or 0)
            dy = float(n.get("dy", -34) if n.get("dy") is not None else -34)
        except Exception:  # noqa: BLE001 - skip a malformed note, keep the rest
            continue
        if not text:
            continue
        # Wrap long notes so one sentence does not stretch across the chart.
        wrapped = "<br>".join(
            text[i : i + 34] for i in range(0, len(text), 34)
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=wrapped,
            showarrow=True,
            arrowhead=2,
            arrowsize=0.9,
            arrowwidth=1,
            arrowcolor=t["muted"],
            ax=dx,
            ay=dy,
            align="left",
            captureevents=True,
            font=dict(size=10, color=t["text"]),
            bgcolor=t["tag_bg"],
            bordercolor=t["muted"],
            borderwidth=1,
            borderpad=4,
            opacity=0.95,
            row=1,
            col=1,
        )


def _add_event_markers(
    fig: go.Figure, events: pd.DataFrame, predicted: pd.DataFrame, t: dict
) -> None:
    """Flag bars in the scenario where indicator events fire.

    Several events often land on one bar, so they are grouped per bar and
    per direction and the hover text lists them all.
    """
    if "_ts" not in events.columns:
        return

    for bias, color, sym, sign in (
        ("bullish", UP, "triangle-up", -1),
        ("bearish", DOWN, "triangle-down", 1),
    ):
        sub = events[events["Bias"] == bias]
        if sub.empty:
            continue

        xs, ys, texts = [], [], []
        for ts, grp in sub.groupby("_ts", sort=True):
            if ts not in predicted.index:
                continue
            bar = predicted.loc[ts]
            span = float(predicted["High"].max() - predicted["Low"].min()) or 1.0
            pad = span * 0.06
            y = float(bar["Low"]) - pad if sign < 0 else float(bar["High"]) + pad
            xs.append(_dt(ts))
            ys.append(y)
            texts.append(
                "<br>".join(f"{r.Indicator}: {r.Event}" for r in grp.itertuples())
            )

        if not xs:
            continue

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"{bias} signal",
                marker=dict(symbol=sym, size=11, color=color,
                            line=dict(width=0.5, color=t["hover_bg"])),
                hovertemplate="%{text}<extra></extra>",
                text=texts,
                showlegend=False,
            ),
            row=1,
            col=1,
        )


def _add_panel(
    fig: go.Figure,
    name: str,
    series_map: dict[str, pd.Series],
    row: int,
    t: dict,
) -> None:
    colors = ["#2f7fd1", "#e0821a", "#9c37b8", "#0f9aa8"]
    # +DI/-DI read as direction, so they get the same green/red as candles
    # rather than an arbitrary palette slot.
    fixed = {"+DI": UP, "-DI": DOWN, "ADX": "#b98cd8"}
    ci = 0

    for label, s in series_map.items():
        if label == "Histogram":
            fig.add_trace(
                go.Bar(
                    x=s.index,
                    y=s,
                    name=label,
                    marker_color=[
                        UP if (v or 0) >= 0 else DOWN for v in s.fillna(0.0)
                    ],
                    opacity=0.55,
                ),
                row=row,
                col=1,
            )
            continue

        dash = "dot" if label.startswith(("Signal", "%D", "QQE trailing")) else None
        key = next((k for k in fixed if label.startswith(k)), None)
        if key:
            color = fixed[key]
            width = 1.6 if key == "ADX" else 1.3
        else:
            color = colors[ci % len(colors)]
            width = 1.3
            ci += 1

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s,
                mode="lines",
                name=label,
                line=dict(width=width, color=color, dash=dash),
                hovertemplate="%{y:.2f}<extra>" + label + "</extra>",
            ),
            row=row,
            col=1,
        )
        _tag_last(fig, s, color, row=row, t=t)

    # Reference levels that make each oscillator readable at a glance.
    def hline(y, color, dash="dash"):
        fig.add_hline(
            y=y, line_width=1, line_dash=dash, line_color=color, row=row, col=1
        )

    if name == "RSI":
        hline(70, DOWN)
        hline(50, t["level"], "dot")
        hline(30, UP)
        fig.update_yaxes(range=[0, 100], dtick=25, row=row, col=1)
    elif name == "Stochastic":
        hline(80, DOWN)
        hline(20, UP)
        fig.update_yaxes(range=[0, 100], dtick=25, row=row, col=1)
    elif name == "QQE":
        hline(50, t["level"], "dot")
        fig.update_yaxes(range=[0, 100], dtick=25, row=row, col=1)
    elif name in ("Price Oscillator", "MACD"):
        hline(0, t["level"], "solid")
    elif name == "DMI / ADX":
        # 25 is Wilder's conventional line between trending and ranging.
        hline(25, t["level"], "dash")
        fig.update_yaxes(rangemode="tozero", row=row, col=1)
