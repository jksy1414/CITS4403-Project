from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]

def runs_dir(model_name="ca") -> Path:
    d = BASE_DIR / "data" / "runs" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d

def figures_dir(model_name="ca") -> Path:
    d = BASE_DIR / "data" / "figures" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d

def timestamped_run_path(model_name="ca", prefix="run", ext=".json") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir(model_name) / f"{prefix}_{ts}{ext}"
