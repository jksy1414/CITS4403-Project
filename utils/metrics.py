from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _as_proportion(x: Any) -> float:
    """
    Treat x as a proportion in [0,1]. If a percentage (>1) slipped in,
    convert it by dividing by 100. Non-numeric or missing → 0.0.
    """
    try:
        val = float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return val / 100.0 if val > 1.0 else val


def summarise(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull basic metrics out of a single run dictionary.
    Assumes keys like I_f/I_r (series), shares_f/shares_r (series),
    reach_fake/reach_real (proportions or percentages).
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    shares_f = run.get("shares_f", []) or []
    shares_r = run.get("shares_r", []) or []

    reach_fake = _as_proportion(run.get("reach_fake", 0.0))
    reach_real = _as_proportion(run.get("reach_real", 0.0))

    peak_f = max(I_f) if I_f else 0
    peak_r = max(I_r) if I_r else 0

    # Keep the same “first occurrence” semantics for t_peak_* as .index()
    t_peak_f = (I_f.index(peak_f) if I_f and (peak_f in I_f) else None)
    t_peak_r = (I_r.index(peak_r) if I_r and (peak_r in I_r) else None)

    return {
        "peak_f": peak_f,
        "peak_r": peak_r,
        "t_peak_f": t_peak_f,
        "t_peak_r": t_peak_r,
        "total_shares_f": int(sum(shares_f)),
        "total_shares_r": int(sum(shares_r)),
        "reach_fake": reach_fake,  # proportions
        "reach_real": reach_real,  # proportions
    }


def time_to_equilibrium(series: List[int], window: int = 10, tol: float = 1e-6) -> Optional[int]:
    """
    Return the earliest t where the rolling mean over `window` samples
    stops changing (difference ≤ tol) compared to the previous window.
    Requires at least 2*window points.
    """
    if not series or len(series) < 2 * window:
        return None

    a = np.asarray(series, dtype=float)
    # Scan forward; identical to the original semantics
    for t in range(window, len(a) - window):
        prev_mean = a[t - window: t].mean()
        curr_mean = a[t: t + window].mean()
        if abs(curr_mean - prev_mean) <= tol:
            return t
    return None


def equilibrium_checks(run: Dict[str, Any], window: int = 10, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Compute equilibrium times for fake/real series in a run.
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    return {
        "t_eq_fake": time_to_equilibrium(I_f, window=window, tol=tol),
        "t_eq_real": time_to_equilibrium(I_r, window=window, tol=tol),
    }
