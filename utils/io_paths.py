from pathlib import Path
from datetime import datetime

# Base directory = project root (one level above 'utils')
BASE_DIR = Path(__file__).resolve().parent.parent

def runs_dir(model_name: str = "ca") -> Path:
    """Return (and create if needed) the directory for run JSON outputs."""
    d = BASE_DIR / "data" / "runs" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d

def figures_dir(model_name: str = "ca") -> Path:
    """Return (and create if needed) the directory for figure outputs."""
    d = BASE_DIR / "data" / "figures" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d

def timestamped_run_path(model_name: str = "ca", prefix: str = "run", ext: str = ".json") -> Path:
    """Return a timestamped path for saving a run result JSON."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir(model_name) / f"{prefix}_{ts}{ext}"
