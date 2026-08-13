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
| Trend & Fibonacci | Swing pivots, the live impulse leg, retracements and projections |
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

### Trends and Fibonacci

Everything hangs off one detected leg, so the pieces agree with each other —
a Fibonacci grid anchored to a different swing than the trend line beside it
is worse than none at all.

1. **Swing pivots** are fractals: a high with *n* lower highs either side.
   The **Swing sensitivity** slider is that *n* — raise it for fewer, larger
   swings.
2. **The impulse leg** runs from the last confirmed pivot to the running
   extreme since. A pivot needs *n* bars of confirmation, so the far end of
   the leg price is working on right now is never itself a confirmed pivot —
   and that is exactly the leg worth measuring. Falling back to two confirmed
   pivots would always draw you a stale leg.
3. **Retracements** run from the end of the leg (`0`) back to its start
   (`1`). **Projections** continue past the end in the leg's direction, so
   `1.618` on a rally from 100 to 200 sits at 261.8.
4. **The trend line** joins the last two pivots of the same kind — lows in an
   uptrend, highs in a downtrend — and extends *forward only*. Running it
   backwards is arithmetically fine and visually useless: a steep line thrown
   back across months leaves the chart and drags the y-axis with it.

The levels are computed from **history alone**, never from your drawn bars,
so they stay put as you redraw — they are the reference you are measuring
against. Crossings inside your scenario show up in **Triggered signals**
alongside the indicator events, so you can ask "does my scenario reach the
1.618 projection?" and get an answer.

### Drawing it yourself

Auto-detection is only the default. The row above the chart offers three
ways to set the anchors:

- **Auto (last impulse)** — the detected swing described above.
- **Choose exact points** — name the bar *and* the price for each end:
  `Leg starts [31 Jul] at its [Low]  →  Leg ends [12 Aug] at its [High]`.
  Defaults land on the auto-detected swing, so you nudge rather than build
  from nothing, and the resulting leg and its size are printed underneath.
  This is the reliable one: no browser interaction involved.
- **Drag a box** — the usual way to place a Fibonacci tool. Drag a
  rectangle from one end of the swing to the other; its height is the leg.
  The box is drawn on the chart so what the levels measure is
  unmistakable, and **Clear** resets it.

  A rectangle carries no direction of its own, so the direction is read
  from the price action inside it: a rally is measured low-to-high, a
  decline high-to-low. Unlike clicking a bar, the box's edges are free
  prices — you are not restricted to a high or a close.

  Selecting this mode switches the chart's drag mode from **pan** to
  **select**, because a pan drag swallows the gesture and Streamlit never
  receives a selection. Scroll still zooms, and the modebar's pan button is
  still there if you need to move around mid-drag.

  Once a box exists, five fields appear to refine it: the two dates, the
  **Bottom** and **Top** prices, and **Shift both**, which slides the whole
  box while keeping its height. The rectangle, the leg line and every level
  are derived from those numbers, so they move as one and cannot drift out
  of agreement — which is also why the box is not draggable in the browser:
  plotly would move the rectangle without telling the app, and the levels
  would stay behind.

The chart modebar also carries Plotly's freehand tools — **line**, **path**,
**rectangle** and an **eraser**. Pick a tool, drag on the chart, and draw
whatever you like: your own trendlines, channels, boxes around a range.

> Freehand shapes are annotations you draw, not inputs: nothing reads their
> coordinates back, so they do not move the Fibonacci levels or appear in
> the signals table. Use **Click two points** for anchors that actually
> drive a calculation. Whether a freehand shape survives a rerun depends on
> Plotly reusing the figure via `uirevision` — verify it in your browser
> before relying on one.

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

Two buttons under the bars table:

- **💾 Save bars** downloads a `.json` to your device.
- **📂 Open bars** loads one back, converted to whatever interval you are
  currently looking at.

That is the whole workflow. The file is yours: it never touches the server,
so it is private to you, works on any computer, and survives every redeploy.
Name it first if you want something friendlier than `XU100.IS-1d.json`.

**Keep a copy on this computer** appears only when you are sitting at the
machine running the app. It writes `scenarios/<name>.json` with an Open and
Delete list — handy for local use, and gated because those files live on the
*server*, not on the viewer's device. Two things have to hold:

- `is_ephemeral()` — false. Streamlit Cloud mounts the repo under
  `/mount/src`, where the folder is shared by every visitor and wiped on
  redeploy. Override with `BARPREDICTION_FORCE_LOCAL=1`.
- The viewer is on loopback. Streamlit serves a Network URL by default, so
  without this anyone on your LAN would get a Load/Delete list for *your*
  scenarios. Satisfied if the server is bound to loopback
  (`--server.address 127.0.0.1`) or the client connected from it.
  `ip_is_local()` fails closed: anything not recognisably loopback counts as
  remote. Force it off entirely with `BARPREDICTION_NO_LOCAL_FILES=1`.

Both routes write the same JSON, so a file saved one way loads the other.

### Changing timeframe converts what you drew

You do not need to save and reload to change timeframe. **Switch the Interval
dropdown and the bars you already drew are converted in place.** Draw three
weekly bars, flip to `1d`, and you have fifteen daily bars carrying the same
shape. Flip back and you get your three weekly bars again, unchanged.

Bars are sticky in general — they survive every setting change except the
ones that mean "start over":

| Change | What happens to your bars |
|---|---|
| Interval | Converted to the new timeframe |
| Horizon slider | Resized — shrinking keeps the front, growing pads flat |
| Indicators, chart display | Untouched |
| Symbol | Reseeded (different instrument) |
| Bar source or its settings | Reseeded (you asked for a new shape) |
| **Reseed bars** button | Reseeded |

Conversion works **both ways, without losing detail**. Coarsening is
genuinely lossy — one weekly bar cannot encode five distinct daily paths, so
naively expanding it back would hand you a straight line instead of the days
you drew. The dashboard therefore remembers the bars at each timeframe and
restores them when you return, as long as you have not edited the coarse
view in between. Edit the weekly bar and going back to daily re-derives from
your edit, rather than silently restoring stale days.

Intraday↔daily is the one conversion that is refused; your bars are left
alone and a message explains why.

Loading a saved scenario works the same way, so a file saved at any interval
opens correctly at any other:

```
first daily open   == the weekly open
last daily close   == the weekly close
max of daily highs == the weekly high
min of daily lows  == the weekly low
```

**Gap fill** decides what the invented bars in between look like:

- **Random walk** (default) — a Brownian bridge pinned to your open and
  close, so the interior wanders like a real session. It is *scaled* to fit
  inside the coarse bar's range rather than clipped against it; clipping
  leaves runs of identical closes that look nothing like a real chart.
  Seeded per bar, so the path is stable across Streamlit reruns and only
  changes when you change the seed.
- **Straight line** — a monotonic path from open to close. Nothing between
  your endpoints moves against the direction you drew.

Changing the mode or the seed **re-draws the gaps in the bars already on
screen** — you do not have to switch interval and back. The coarse bars the
upsample came from are kept for exactly this. If you have hand-edited the
finer bars since, they are left alone and a message says so, because
re-filling would discard your edits.

Either way the bars you drew are reproduced exactly, and nothing escapes
their range. Going the other direction (daily → weekly) is ordinary OHLC
aggregation, and a short final group still becomes a bar.

Ratios are in trading bars, not calendar time: a week is 5 sessions, a month
21. Intraday↔daily conversion is **refused** rather than guessed, because the
ratio depends on session length.

## Reading the chart

Laid out along the lines trading platforms use
([TradingView's chart settings](https://www.tradingview.com/support/solutions/43000748166-how-to-configure-your-supercharts/)
were the reference):

- A **status line** top-left — symbol, interval, and the last bar's O/H/L/C
  with the change, tinted green or red. It shows the last bar rather than
  the hovered one: hover stays in the browser and never reaches the server.
- **Indicator legend** on the row beneath it.
- **Empty space past the last bar**, about 9% of the visible span, so the
  newest bar is not jammed against the price scale and there is room to
  project a level or place a note ahead of price.
- A faint **watermark** of symbol and interval behind the candles.
- **Price scale on the right**, with a value badge per series.

**Price style** sits beside the indicator pills, above the chart:
**Candles**, **OHLC bars** or **Line**.

**Notes on the chart** — under the bars table. Type the text, pick the bar
and the price, press **Add**, and it appears as a labelled box with an arrow
pointing at that spot. Long notes wrap.

Notes are listed in an editable table: change the text, the bar it points
at, the price, or the **Nudge** offsets that push the box away from that
point. Add and remove rows there too.

> You can also **drag a note** on the chart, along with any freehand shape.
> That is a quick visual nudge only — Plotly does not report the new
> position back to the app, so it resets on the next redraw. Set the Nudge
> columns to make a position permanent.
>
> Reference lines are **locked**: RSI's 50, Stochastic's 20/80, DMI's 25,
> the Fibonacci levels and the scenario shading cannot be dragged. A
> reference level nudged off its own value would quietly lie about what it
> marks. Only shapes you draw yourself can be moved.

Notes are saved and loaded with the scenario, offsets included, so a
downloaded file carries your annotations exactly as you placed them. All three show the same bars — OHLC draws open and
close as ticks either side of the range, Line plots closes only.

- Prices sit on the **right**, trading-platform style.
- Each line gets a **badge** on the right axis showing its current value.
- **Triangles** mark bars where indicator events fire — green/up for bullish,
  red/down for bearish. Hover one to see every event on that bar.
- The scenario region is **shaded**, so drawn bars never read as real history.
- **Zoom the price axis on its own** by dragging it up or down;
  double-click to autoscale. Scroll zooms both axes, drag pans.
- **Indicator panes are locked** and autoscale themselves. Dragging an
  oscillator's axis would pull it off the range its levels are defined
  against — an RSI stretched past 0–100 puts its own 30 and 70 lines
  somewhere meaningless.

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
