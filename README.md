# BIST bar prediction

A what-if tool. Pick a BIST symbol, propose the next few bars, then switch
indicators on and off to see what your scenario does to them.

The question it answers: *"if THYAO closes here for the next three days, does
my QQE flip? does SAR turn? does RSI break 70?"*

```
python -m streamlit run app.py
```

Then open http://localhost:8501.

## How it works

1. **Load** history from Yahoo (BIST tickers use the `.IS` suffix, e.g.
   `THYAO.IS`, `XU100.IS`) or upload your own OHLC CSV.
2. **Propose** the next N bars from a *bar source* — hand-drawn, a ramp, or a
   generated one. Every source is editable: the table on the left takes your
   numbers and the chart updates.
3. **Inspect** by ticking indicators in the sidebar. They compute over history
   plus your bars as one continuous series, so lines run straight through the
   boundary rather than restarting.

Three tabs under the chart:

- **Triggered signals** — events that fire *inside* your predicted bars (SAR
  flip, %K/%D cross, RSI level breaks, QQE trend flip, MA crossovers, PPO
  zero/signal crosses).
- **Before / after** — every indicator's value at the last real bar vs the
  last predicted bar, with the delta.
- **Indicator values** — the raw table, downloadable as CSV.

## Indicators

| Indicator | Notes |
|---|---|
| Parabolic SAR | Wilder's, with AF start/increment/max configurable |
| Moving averages | Stack up to four; SMA / EMA / WMA / RMA, any period |
| RSI | Wilder smoothing (RMA), not a plain EMA |
| Stochastic | Slow by default; set %K smoothing to 1 for fast |
| Price Oscillator | PPO (percentage) or absolute, with signal line and histogram |
| QQE | Smoothed RSI plus its volatility-scaled trailing stop |

RSI and QQE use Wilder's RMA because that is what their authors specified — an
EMA of the same period gives visibly different numbers and would not line up
with a TradingView or MetaTrader chart.

## Adding a different prediction style

Bar sources are pluggable. The whole contract is one method:

```python
from core.predictors.base import Param, Predictor

class MyModel(Predictor):
    name = "Auto - my model"
    description = "What it does."
    params = (Param("lookback", "Lookback bars", 60, "int", min=10, max=500),)

    def propose(self, history, index):
        # history: OHLCV DataFrame. index: the future timestamps to fill.
        rows = [...]                      # one dict per bar
        return self._frame(index, rows)   # validates and returns
```

Register it in `core/predictors/__init__.py` by adding the class to
`_SOURCES`. The dashboard picks it up automatically and renders a control for
each `Param`. Nothing else changes — this is where an ARIMA, gradient boosting,
or sequence model would slot in.

## Layout

```
app.py                     Streamlit dashboard
core/
  data.py                  Yahoo + CSV loading, future bar index
  indicators.py            All six indicators, from their definitions
  charting.py              Plotly candlestick + oscillator panels
  signals.py               Event detection inside the predicted window
  predictors/
    base.py                Predictor interface, OHLC sanitising
    manual.py              Hand-drawn seeds (flat, ramp)
    auto.py                Drift, random walk, pattern replay
tests/test_indicators.py   Reference-value and invariant checks
```

Run the checks with `python tests/test_indicators.py`.

## If the chart feels slow

Panning and zooming happen in the browser, so smoothness depends on how much
the figure has to redraw. In rough order of impact:

- **History bars shown** (sidebar, Display) is the big one — candlesticks are
  the expensive trace and cost scales directly with bar count. 150 is the
  default; 600 is the cap.
- **Crosshair** draws spike lines across every panel on each mouse move. Off
  by default; turn it on when you need to read values across panels.
- **Unified hover** gathers every series into one tooltip. Off by default.
- Fewer indicators means fewer panels to redraw.

Pan and zoom survive reruns (`uirevision`), so editing a bar no longer snaps
the view back to full extent. Double-click resets it deliberately.

Line traces are plain SVG rather than WebGL on purpose: at these point counts
WebGL gains nothing measurable, and it renders as *nothing at all* anywhere a
WebGL context is unavailable.

## Notes and caveats

- **Mid-session bars.** While BIST is open, the newest bar is still forming.
  Fully-empty placeholder bars are dropped automatically; a live partial bar
  is kept, since it has real data. Tick **Drop the final bar** in the sidebar
  to exclude it and work from completed bars only.
- **Intraday history is short.** Yahoo caps 5m/15m/30m at 60 days and 60m at
  two years; the loader clamps the request for you.
- **The auto sources are not forecasts.** A drift line and a random walk are
  there to give you a starting shape and to demonstrate the plug-in point.
  Treat the output as a scenario you chose to examine, not a prediction of
  where BIST is going.
