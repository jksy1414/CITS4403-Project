# src/ca/rules.py
from __future__ import annotations
import numpy as np
from typing import Optional
from .states import State, Params

def update_cell(
    curr: int,
    saw_f: bool,
    saw_r: bool,
    rng: np.random.Generator,
    p: Params,
    *,
    share_f_prob: Optional[float] = None,
    share_r_prob: Optional[float] = None,
) -> int:
    """
    Per-cell transition. Supports optional precomputed share probabilities.
    If share_f_prob/share_r_prob are None, they are derived from Params.
    """
    # --- defaults if not provided
    if share_f_prob is None:
        share_f_prob = p.beta_share_f
    if share_r_prob is None:
        share_r_prob = p.beta_share_r

    # correction dampens sharing fake if real was seen
    if saw_r:
        share_f_prob *= (1.0 - p.gamma_correction)

    # Safety clamps (hetero multipliers may push slightly out of bounds)
    share_f_prob = float(np.clip(share_f_prob, 0.0, 1.0))
    share_r_prob = float(np.clip(share_r_prob, 0.0, 1.0))

    s = curr

    if s == State.S:
        if saw_f and not saw_r:
            return State.E_F
        if saw_r and not saw_f:
            return State.E_R
        if saw_f and saw_r:
            # tie-break bias towards real (60/40)
            return State.E_R if rng.random() < 0.6 else State.E_F
        return s

    if s == State.E_F:
        if saw_f and (rng.random() < share_f_prob):
            return State.I_F
        # tiny chance to move to E_R if real is visible
        if saw_r and (rng.random() < p.gamma_switch * 0.2):
            return State.E_R
        # forgetting
        if rng.random() < p.delta_decay_f:
            return State.S
        return s

    if s == State.E_R:
        if saw_r and (rng.random() < share_r_prob):
            return State.I_R
        if rng.random() < p.delta_decay_r:
            return State.S
        return s

    if s == State.I_F:
        if saw_r and (rng.random() < p.gamma_switch):
            return State.I_R
        if rng.random() < p.delta_decay_f:
            return State.E_F
        return s

    if s == State.I_R:
        if rng.random() < p.delta_decay_r:
            return State.E_R
        return s

    return s
