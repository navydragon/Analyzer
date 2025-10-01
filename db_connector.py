from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DB = Path('.local_results.json')


@dataclass
class SavedRun:
    ts: float
    kind: str
    payload: dict[str, Any]


def save_result(kind: str, payload: dict[str, Any]) -> None:
    items: list[dict[str, Any]] = []
    if _DB.exists():
        items = json.loads(_DB.read_text(encoding='utf-8'))
    items.append(asdict(SavedRun(ts=time.time(), kind=kind, payload=payload)))
    _DB.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
