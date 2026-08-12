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
    },
}


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
    pickable: pd.Series | None = None,
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
        go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            name="History",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
            increasing_fillcolor=UP,
            decreasing_fillcolor=DOWN,
            line_width=1,
        ),
        row=1,
        col=1,
    )

    if not predicted.empty:
        fig.add_trace(
            go.Candlestick(
                x=predicted.index,
                open=predicted["Open"],
                high=predicted["High"],
                low=predicted["Low"],
                close=predicted["Close"],
                name="Your bars",
                increasing_line_color=PRED_UP,
                decreasing_line_color=PRED_DOWN,
                increasing_fillcolor="rgba(102,187,106,0.45)",
                decreasing_fillcolor="rgba(255,112,67,0.45)",
                line_width=1,
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

    if pickable is not None and not pickable.empty:
        # Candlesticks do not report clicks usefully, so a marker per bar
        # sits on top to catch them; Streamlit reads the selection back and
        # turns it into Fibonacci anchors. These are deliberately visible:
        # a fully transparent marker is neither clickable nor discoverable,
        # so there is nothing to aim at.
        fig.add_trace(
            go.Scatter(
                x=pickable.index,
                y=pickable,
                mode="markers",
                name="click to anchor",
                marker=dict(
                    size=7,
                    color="rgba(215,225,240,0.35)",
                    line=dict(width=1, color="rgba(120,170,220,0.85)"),
                ),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra>click to anchor</extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    _add_overlays(fig, overlays)

    if overlay_shapes:
        _add_trend_and_fib(fig, overlay_shapes, t)

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
        margin=dict(l=4, r=96, t=52, b=4),
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
            y=1.012,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=t["text"]),
            itemsizing="constant",
        ),
        template=t["template"],
        font=dict(color=t["text"], size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
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
