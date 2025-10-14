from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


def summarise(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract core summary statistics from a single simulation run.
    Includes peaks, timing, total shares, and overall reach for
    both fake and real information series.
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    shares_f = run.get("shares_f", []) or []
    shares_r = run.get("shares_r", []) or []

    reach_fake = float(run.get("reach_fake", 0.0) or 0.0)
    reach_real = float(run.get("reach_real", 0.0) or 0.0)

    # Convert percentages (>1) into proportions (0–1)
    if reach_fake > 1.0:
        reach_fake /= 100.0
    if reach_real > 1.0:
        reach_real /= 100.0

    peak_f = max(I_f) if I_f else 0
    peak_r = max(I_r) if I_r else 0

    t_peak_f = (I_f.index(peak_f) if I_f and peak_f in I_f else None)
    t_peak_r = (I_r.index(peak_r) if I_r and peak_r in I_r else None)

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


def time_to_equilibrium(series: List[int], window: int = 10, tol: float = 1e-6) -> Optional[int]:
    """
    Identify when a time series stabilises by comparing consecutive
    rolling window averages. Returns the index (time step) where
    equilibrium first occurs, or None if stability is never reached.
    """
    if not series or len(series) < 2 * window:
        return None

    arr = np.asarray(series, dtype=float)
    for t in range(window, len(arr) - window):
        prev_mean = arr[t - window:t].mean()
        curr_mean = arr[t:t + window].mean()
        if abs(curr_mean - prev_mean) <= tol:
            return t
    return None


def equilibrium_checks(run: Dict[str, Any], window: int = 10, tol: float = 1e-6) -> Dict[str, Optional[int]]:
    """
    Run the stability test for both fake and real information series.
    Returns the time step where each stabilises (if any).
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []

    return {
        "t_eq_fake": time_to_equilibrium(I_f, window=window, tol=tol),
        "t_eq_real": time_to_equilibrium(I_r, window=window, tol=tol),
    }


# --- Optional backward-compatible aliases (safe for older imports) ---
extract_run_metrics = summarise
detect_stable_region = time_to_equilibrium
estimate_equilibrium_timeseries = equilibrium_checks
