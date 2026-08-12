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

    _add_overlays(fig, overlays)

    for i, (name, series_map) in enumerate(panels, start=2):
        _add_panel(fig, name, series_map, row=i, t=t)

    fig.update_layout(
        height=total,
        margin=dict(l=4, r=4, t=52, b=4),
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
    )
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


def _add_panel(
    fig: go.Figure,
    name: str,
    series_map: dict[str, pd.Series],
    row: int,
    t: dict,
) -> None:
    colors = ["#2f7fd1", "#e0821a", "#9c37b8", "#0f9aa8"]
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
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s,
                mode="lines",
                name=label,
                line=dict(width=1.3, color=colors[ci % len(colors)], dash=dash),
                hovertemplate="%{y:.2f}<extra>" + label + "</extra>",
            ),
            row=row,
            col=1,
        )
        ci += 1

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
    elif name == "Price Oscillator":
        hline(0, t["level"], "solid")
