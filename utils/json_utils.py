from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


class NumpySafeEncoder(json.JSONEncoder):
    """
    JSON encoder that gracefully handles NumPy values and Path objects.
    Converts them into plain Python types so they can be serialized safely.
    """
    def default(self, obj: Any) -> Any:
        # Handle all NumPy numeric scalars (int32, float64, etc.)
        if isinstance(obj, np.generic):
            return obj.item()
        # Convert entire arrays to lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Represent file system paths as strings
        if isinstance(obj, Path):
            return str(obj)
        # Fallback to default encoder
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Save a Python dictionary to a JSON file.
    Automatically creates directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            cls=NumpySafeEncoder
        )


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load and parse a JSON file into a Python dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
