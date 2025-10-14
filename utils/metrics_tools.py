from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

def extract_run_metrics(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate basic metrics from simulation time series:
    - Peak posters
    - Time to peak
    - Total shares
    - Reach (as proportion)
    """
    I_f = run_data.get("I_f", []) or []
    I_r = run_data.get("I_r", []) or []
    shares_f = run_data.get("shares_f", []) or []
    shares_r = run_data.get("shares_r", []) or []

    reach_fake = float(run_data.get("reach_fake", 0.0) or 0.0)
    reach_real = float(run_data.get("reach_real", 0.0) or 0.0)
    if reach_fake > 1.0: reach_fake /= 100.0
    if reach_real > 1.0: reach_real /= 100.0

    peak_f = max(I_f, default=0)
    peak_r = max(I_r, default=0)

    t_peak_f = I_f.index(peak_f) if peak_f in I_f else None
    t_peak_r = I_r.index(peak_r) if peak_r in I_r else None

    return {
        "peak_f": peak_f,
        "peak_r": peak_r,
        "t_peak_f": t_peak_f,
        "t_peak_r": t_peak_r,
        "total_shares_f": int(sum(shares_f)),
        "total_shares_r": int(sum(shares_r)),
        "reach_fake": reach_fake,
        "reach_real": reach_real,
    }

def detect_stable_region(values: List[int], window_size: int = 10, tolerance: float = 1e-6) -> Optional[int]:
    """
    Estimate when a series stabilizes by comparing rolling mean windows.
    Returns the first index where stability is detected.
    """
    if len(values) < 2 * window_size:
        return None

    array = np.array(values, dtype=float)
    for t in range(window_size, len(array) - window_size):
        prev_avg = array[t - window_size: t].mean()
        curr_avg = array[t: t + window_size].mean()
        if abs(curr_avg - prev_avg) <= tolerance:
            return t
    return None

def estimate_equilibrium_timeseries(run_data: Dict[str, Any], window_size: int = 10, tolerance: float = 1e-6) -> Dict[str, Optional[int]]:
    """
    Run stability check on fake and real time series independently.
    """
    I_f = run_data.get("I_f", []) or []
    I_r = run_data.get("I_r", []) or []
    return {
        "t_eq_fake": detect_stable_region(I_f, window_size, tolerance),
        "t_eq_real": detect_stable_region(I_r, window_size, tolerance),
    }
