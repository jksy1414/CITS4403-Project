from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_output_path(subfolder: str, model: str = "ca") -> Path:
    
    path = PROJECT_ROOT / "data" / subfolder / model
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_run_output_dir(model: str = "ca") -> Path:
    
    return get_output_path("runs", model)

def get_figure_output_dir(model: str = "ca") -> Path:
    
    return get_output_path("figures", model)

def generate_timestamped_filename(prefix: str = "run", model: str = "ca", ext: str = ".json") -> Path:
    
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}{ext}"
    return get_run_output_dir(model) / filename

