from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict
import numpy as np

class JSONSafeEncoder(json.JSONEncoder):
    """
    Custom encoder that safely converts NumPy and Path object, 
    convert non-standard object to standard Python types.
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def export_to_json(data: Dict[str, Any], destination: Path) -> None:
    """
    Save a dic to JSON format, ensuring directories exist, can handle NumPY types.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, cls=JSONSafeEncoder)


def import_from_json(source: Path) -> Dict[str, Any]:
    """
    Load a JSON file and return the contents as a dic.
    """
    with source.open("r", encoding="utf-8") as file:
        return json.load(file)
