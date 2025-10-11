from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any
import json

def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_run(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_runs(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for x in paths:
        p = Path(x)
        if p.exists():
            out.append(load_run(p))
    return out

def find_runs(folder: str | Path, pattern: str = "CA_run_*.json", recursive: bool = False) -> List[Path]:
    base = Path(folder)
    paths = sorted(base.rglob(pattern) if recursive else base.glob(pattern))
    return [p for p in paths if p.is_file()]
