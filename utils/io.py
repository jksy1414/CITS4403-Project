from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder that safely converts NumPy and Path types to JSON-friendly forms."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save Python dict as a nicely formatted JSON file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file and return as Python dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
