"""Save and reload hand-drawn bar scenarios.

Two routes, because they suit different situations:

**Download / upload** hands the JSON to the browser, so the file lands on
whichever device is using the app. This is the one that works on a hosted
deployment: nothing is kept server-side, so scenarios are private to the
person who made them and survive every redeploy.

**Local files** under `scenarios/` are a convenience for running the app on
your own machine. On a shared host that directory is one folder shared by
every visitor *and* wiped on restart, so the dashboard hides it there --
see `is_ephemeral`.

Both routes use the same JSON shape, so a file saved one way loads the other.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DIR = Path(__file__).resolve().parents[1] / "scenarios"

FORMAT = 1


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


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


LOOPBACK = frozenset(
    {"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1", "::ffff:127.0.0.1"}
)


def ip_is_local(ip: object) -> bool:
    """Is this client address the machine serving the page?

    Fails closed. Anything that is not a recognised loopback string -- a LAN
    address, None, or some object a test harness substituted -- counts as
    remote, because the cost of guessing wrong is exposing one person's
    saved scenarios to another.
    """
    if not isinstance(ip, str):
        return False
    return ip.strip().lower() in LOOPBACK


def is_ephemeral() -> bool:
    """True when server-side files are shared between users or won't survive.

    Streamlit Community Cloud mounts the repo under /mount/src and rebuilds
    the container on redeploy, so `scenarios/` there is neither private nor
    durable. Set BARPREDICTION_FORCE_LOCAL=1 to override.
    """
    if os.environ.get("BARPREDICTION_FORCE_LOCAL") == "1":
        return False
    if os.environ.get("BARPREDICTION_EPHEMERAL") == "1":
        return True
    return str(DIR).replace("\\", "/").startswith("/mount/src")


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def _payload(name: str, symbol: str, interval: str, bars: pd.DataFrame) -> dict:
    return {
        "format": FORMAT,
        "name": name.strip(),
        "symbol": symbol,
        "interval": interval,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bars": [
            {"ts": str(ts), **{c: float(row[c]) for c in COLUMNS}}
            for ts, row in bars[COLUMNS].iterrows()
        ],
    }


def _parse(raw: dict, fallback_name: str = "scenario") -> Scenario:
    rows = raw["bars"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("file contains no bars")

    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"bars are missing {missing}")

    df = pd.DataFrame(
        [{c: float(r[c]) for c in COLUMNS} for r in rows],
        index=pd.DatetimeIndex([pd.Timestamp(r["ts"]) for r in rows]),
    )
    df.index.name = "Date"
    return Scenario(
        name=str(raw.get("name") or fallback_name),
        symbol=str(raw.get("symbol", "?")),
        interval=str(raw.get("interval", "1d")),
        saved_at=str(raw.get("saved_at", "")),
        bars=df,
    )


def _validate(name: str, bars: pd.DataFrame) -> None:
    if not name.strip():
        raise ValueError("Give the scenario a name.")
    if bars.empty:
        raise ValueError("Nothing to save -- there are no bars.")


def to_json_bytes(
    name: str, symbol: str, interval: str, bars: pd.DataFrame
) -> bytes:
    """Serialise for st.download_button -- the file goes to the user's device."""
    _validate(name, bars)
    return json.dumps(_payload(name, symbol, interval, bars), indent=2).encode("utf-8")


def from_json_bytes(data: bytes, fallback_name: str = "uploaded") -> Scenario:
    """Parse an uploaded scenario file, with a readable error if it is not one."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"not valid JSON ({e})") from None
    if not isinstance(raw, dict) or "bars" not in raw:
        raise ValueError("not a scenario file -- no 'bars' key")
    return _parse(raw, fallback_name)


def filename_for(name: str, symbol: str, interval: str) -> str:
    return f"{_slug(name)}_{_slug(symbol)}_{interval}.json"


# --------------------------------------------------------------------------
# local files
# --------------------------------------------------------------------------


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return (s or "scenario").lower()[:60]


def _path(name: str) -> Path:
    return DIR / f"{_slug(name)}.json"


def save(name: str, symbol: str, interval: str, bars: pd.DataFrame) -> Path:
    """Write a scenario to disk, overwriting any file with the same slug."""
    _validate(name, bars)
    DIR.mkdir(parents=True, exist_ok=True)
    p = _path(name)
    p.write_text(
        json.dumps(_payload(name, symbol, interval, bars), indent=2), encoding="utf-8"
    )
    return p


def _read(p: Path) -> Scenario | None:
    try:
        return _parse(json.loads(p.read_text(encoding="utf-8")), p.stem)
    except Exception:  # noqa: BLE001 - a corrupt file must not break the app
        return None


def load(name: str) -> Scenario:
    p = _path(name)
    sc = _read(p) if p.exists() else None
    if sc is None:
        raise ValueError(f"Could not read scenario {name!r}.")
    return sc


def list_all() -> list[Scenario]:
    """Every readable scenario on disk, newest first."""
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
