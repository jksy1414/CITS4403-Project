from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _convert_for_json(obj: Any) -> Any:
    """
    Best-effort conversion for a few types that the default JSON encoder
    doesn't handle out of the box.
    - NumPy integers -> int
    - NumPy floats   -> float
    - NumPy arrays   -> list
    - pathlib.Path   -> str
    If the object isn't one of the above, return it unchanged.
    """
    # numpy scalar numbers
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # filesystem paths
    if isinstance(obj, Path):
        return str(obj)

    # not something we know how to convert here
    return obj


class NumpyEncoder(json.JSONEncoder):
    """
    JSONEncoder that leans on `_convert_for_json` to serialize a few
    otherwise-problematic Python/NumPy types without changing behaviour.
    """

    def default(self, obj: Any) -> Any:
        converted = _convert_for_json(obj)
        # If we didn't convert it, defer to the base implementation.
        # Otherwise, return the transformed value.
        if converted is obj:
            return super().default(obj)
        return converted


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Write `obj` to `path` as human-readable JSON (indent=2).
    Creates parent folders when they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Read JSON from `path` and return the resulting dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
