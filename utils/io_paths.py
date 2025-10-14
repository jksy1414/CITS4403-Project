from pathlib import Path
from datetime import datetime

# Project root (folder above this file's directory).
BASE_DIR = Path(__file__).resolve().parent.parent


def _subdir(*parts: str) -> Path:
    """
    Join BASE_DIR with `parts`, ensure the folder exists, and return it.
    Keeping this tiny helper avoids repeating mkdir logic.
    """
    d = BASE_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(model_name: str = "ca") -> Path:
    """
    Directory for run outputs (e.g., JSON/CSV produced by simulations).
    Creates the folder if it doesn't already exist.
    """
    return _subdir("data", "runs", model_name)


def figures_dir(model_name: str = "ca") -> Path:
    """
    Directory for saved figures/plots.
    Creates the folder if it doesn't already exist.
    """
    return _subdir("data", "figures", model_name)


def timestamped_run_path(
    model_name: str = "ca",
    prefix: str = "run",
    ext: str = ".json",
) -> Path:
    """
    Build a timestamped file path for a run result, e.g.:
    data/runs/<model_name>/<prefix>_YYYYMMDD-HHMMSS<ext>
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir(model_name) / f"{prefix}_{ts}{ext}"
