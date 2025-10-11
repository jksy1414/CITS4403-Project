from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

def summarise(run: Dict[str, Any]) -> Dict[str, Any]:
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    shares_f = run.get("shares_f", []) or []
    shares_r = run.get("shares_r", []) or []
    reach_fake = float(run.get("reach_fake", 0.0) or 0.0)
    reach_real = float(run.get("reach_real", 0.0) or 0.0)

    peak_f = max(I_f) if I_f else 0
    peak_r = max(I_r) if I_r else 0
    t_peak_f = (I_f.index(peak_f) if I_f and peak_f in I_f else None)
    t_peak_r = (I_r.index(peak_r) if I_r and peak_r in I_r else None)

    return {
        "seed_used": run.get("seed_used"),
        "peak_fake": int(peak_f),
        "peak_real": int(peak_r),
        "t_peak_fake": t_peak_f,
        "t_peak_real": t_peak_r,
        "reach_fake": round(reach_fake * 100, 2),  # %
        "reach_real": round(reach_real * 100, 2),  # %
        "cum_shares_fake": int(sum(shares_f)),
        "cum_shares_real": int(sum(shares_r)),
    }

def time_to_equilibrium(series: List[int], window: int = 10, tol: float = 1e-6) -> Optional[int]:
    if not series or len(series) < 2 * window:
        return None
    a = np.asarray(series, dtype=float)
    for t in range(window, len(a) - window):
        prev_mean = a[t - window: t].mean()
        curr_mean = a[t: t + window].mean()
        if abs(curr_mean - prev_mean) <= tol:
            return t
    return None

def equilibrium_checks(run: Dict[str, Any], window: int = 10, tol: float = 1e-6) -> Dict[str, Any]:
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    return {
        "t_eq_fake": time_to_equilibrium(I_f, window=window, tol=tol),
        "t_eq_real": time_to_equilibrium(I_r, window=window, tol=tol),
    }
