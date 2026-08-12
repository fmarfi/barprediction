"""Save and reload hand-drawn bar scenarios.

One JSON file per scenario under `scenarios/`, holding the bars plus the
symbol and interval they were drawn at. The interval matters: reloading a
weekly scenario onto a daily chart runs it through core.resample first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DIR = Path(__file__).resolve().parents[1] / "scenarios"


@dataclass(frozen=True)
class Scenario:
    name: str
    symbol: str
    interval: str
    saved_at: str
    bars: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{self.name}  ·  {self.symbol} {self.interval} · {len(self.bars)} bars"


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return (s or "scenario").lower()[:60]


def _path(name: str) -> Path:
    return DIR / f"{_slug(name)}.json"


def save(name: str, symbol: str, interval: str, bars: pd.DataFrame) -> Path:
    """Write a scenario, overwriting any file with the same slug."""
    if not name.strip():
        raise ValueError("Give the scenario a name.")
    if bars.empty:
        raise ValueError("Nothing to save -- there are no bars.")

    DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name.strip(),
        "symbol": symbol,
        "interval": interval,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bars": [
            {
                "ts": str(ts),
                **{c: float(row[c]) for c in COLUMNS},
            }
            for ts, row in bars[COLUMNS].iterrows()
        ],
    }
    p = _path(name)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _read(p: Path) -> Scenario | None:
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        rows = raw["bars"]
        df = pd.DataFrame(
            [{c: float(r[c]) for c in COLUMNS} for r in rows],
            index=pd.DatetimeIndex([pd.Timestamp(r["ts"]) for r in rows]),
        )
        df.index.name = "Date"
        return Scenario(
            name=str(raw.get("name", p.stem)),
            symbol=str(raw.get("symbol", "?")),
            interval=str(raw.get("interval", "1d")),
            saved_at=str(raw.get("saved_at", "")),
            bars=df,
        )
    except Exception:  # noqa: BLE001 - a corrupt file must not break the app
        return None


def load(name: str) -> Scenario:
    p = _path(name)
    sc = _read(p) if p.exists() else None
    if sc is None:
        raise ValueError(f"Could not read scenario {name!r}.")
    return sc


def list_all() -> list[Scenario]:
    """Every readable scenario, newest first."""
    if not DIR.exists():
        return []
    out = [sc for sc in (_read(p) for p in DIR.glob("*.json")) if sc is not None]
    return sorted(out, key=lambda s: s.saved_at, reverse=True)


def delete(name: str) -> bool:
    p = _path(name)
    if p.exists():
        p.unlink()
        return True
    return False
