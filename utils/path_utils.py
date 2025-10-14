from pathlib import Path
from datetime import datetime


# The root of the project (one level up from 'utils/')
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_subdir(folder: str, model: str = "ca") -> Path:
    """
    Internal helper that ensures a directory exists under 'data/'.
    Example: data/runs/ca or data/figures/ca
    """
    path = PROJECT_ROOT / "data" / folder / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_output_dir(model: str = "ca") -> Path:
    """Return the directory for simulation run results (JSON/CSV)."""
    return _ensure_subdir("runs", model)


def get_figure_output_dir(model: str = "ca") -> Path:
    """Return the directory for saving generated figures."""
    return _ensure_subdir("figures", model)


def generate_timestamped_filename(prefix: str = "run", model: str = "ca", ext: str = ".json") -> Path:
    """
    Create a timestamped filename under the model's run directory.
    Example output: data/runs/ca/run_20251014-172301.json
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}{ext}"
    return get_run_output_dir(model) / filename


# --- Optional backward-compatibility aliases (safe to keep if old code uses them) ---

runs_dir = get_run_output_dir
figures_dir = get_figure_output_dir
timestamped_run_path = generate_timestamped_filename
