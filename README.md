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
| MACD | Absolute EMA spread, with signal line and coloured histogram |
| DMI / ADX | Wilder's +DI, -DI and ADX, with the conventional 25 line |
| Stochastic | Slow by default; set %K smoothing to 1 for fast |
| Price Oscillator | PPO (percentage) or absolute, with signal line and histogram |
| QQE | Smoothed RSI plus its volatility-scaled trailing stop |

RSI, QQE and DMI use Wilder's RMA because that is what their authors
specified — an EMA of the same period gives visibly different numbers and
would not line up with a TradingView or MetaTrader chart.

**MACD vs Price Oscillator** are the same measurement on different scales:
MACD is `EMA(fast) - EMA(slow)` in price units, PPO divides that by the slow
EMA and reports a percentage. Use PPO to compare symbols trading at different
price levels; a 200-point MACD means something very different on XU100 at
13,700 than on a 30-lira stock. `test_our_ppo_is_macd_normalised` pins the
relationship.

### Are they correct?

Two suites, both runnable without a network:

```
python tests/test_indicators.py        # 18 reference-value and invariant checks
python tests/test_cross_validation.py  # 8 checks against the independent `ta` library
```

The cross-validation compares against [`ta`](https://pypi.org/project/ta/), a
separate implementation by a different author. MACD, Stochastic, ATR and the
PPO/MACD identity agree to **0.00e+00**; ADX correlates at **1.0000** and
+DI/-DI dominance agrees on **100%** of bars; Parabolic SAR lands within 1% on
**99.4%** (it is path dependent, so a different seed direction on bar one can
shift a whole leg).

RSI deliberately does *not* match `ta`. Wilder seeds the average gain/loss
with an SMA of the first `length` changes; `ta` starts its EWM at bar one. On
the StockCharts reference series Wilder's hand-computed first value is
`70.46413502109705` — we return exactly that, `ta` returns `71.80`. The gap is
a seeding artefact that decays to ~3e-08 by bar 250, and both behaviours are
pinned by tests.

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

## Saving scenarios and switching timeframes

Under the bars table, **Save / load this scenario**:

- **⬇ Download bars (.json)** puts the file on *your* device. This is the
  route that works on a hosted deployment — nothing is stored on the server,
  so your scenarios are private to you and survive every redeploy. Restore
  with the uploader beside it, on any machine and at any timeframe.
- **💾 Save to this machine** writes `scenarios/<name>.json`, with a Load and
  Delete list beside it. Convenient when you run the app locally.

> The local-file section is **hidden on a hosted deployment**. There the
> server's disk is one folder shared by every visitor — anyone could load or
> delete anyone else's scenarios — and it is wiped on redeploy. `is_ephemeral()`
> detects this (the repo mounts under `/mount/src` on Streamlit Cloud);
> override with `BARPREDICTION_FORCE_LOCAL=1`.

Both routes write the same JSON, so a file saved one way loads the other.

Because the interval is stored, a scenario drawn on one timeframe can be
loaded onto another. Draw three weekly bars, switch the chart to `1d`, load
it back, and each week is expanded into five daily bars that still add up to
what you drew:

```
first daily open   == the weekly open
last daily close   == the weekly close
max of daily highs == the weekly high
min of daily lows  == the weekly low
```

The closes in between walk a straight line from open to close, so filling the
gaps never invents swings you did not draw. `test_upsampled_closes_are_monotonic`
enforces that; `test_round_trip_weekly_daily_weekly_is_identity` checks the
conversion is lossless both ways. Going the other direction (daily → weekly)
is ordinary OHLC aggregation, and a short final group still becomes a bar.

Ratios are in trading bars, not calendar time: a week is 5 sessions, a month
21. Intraday↔daily conversion is **refused** rather than guessed, because the
ratio depends on session length.

## Reading the chart

- Prices sit on the **right**, trading-platform style.
- Each line gets a **badge** on the right axis showing its current value.
- **Triangles** mark bars where indicator events fire — green/up for bullish,
  red/down for bearish. Hover one to see every event on that bar.
- The scenario region is **shaded**, so drawn bars never read as real history.
- **Zoom the y axis on its own** by dragging the price axis up or down;
  double-click to autoscale. Scroll zooms both axes, drag pans.

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
