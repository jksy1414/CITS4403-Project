# src/ca/rules.py
import numpy as np
from .states import State, Params

def update_cell(
    curr: int,
    saw_f: bool,
    saw_r: bool,
    rng: np.random.Generator,
    p: Params,
    share_f_prob: float,
    share_r_prob: float,
) -> int:
    """
    Pure local rule. We pass the *effective* share probabilities already
    scaled by heterogeneity/spatial multipliers, so there are no None errors.
    """
    s = curr

    # seeing real content reduces fake sharing chance (correction)
    eff_share_f = share_f_prob
    if saw_r:
        eff_share_f *= (1.0 - p.gamma_correction)

    eff_share_r = share_r_prob

    # --- state transitions (same logic as before)
    if s == State.S:
        if saw_f and not saw_r:
            return State.E_F
        if saw_r and not saw_f:
            return State.E_R
        if saw_f and saw_r:
            return State.E_R if rng.random() < 0.6 else State.E_F
        return s

    if s == State.E_F:
        if saw_f and (rng.random() < eff_share_f):
            return State.I_F
        if saw_r and (rng.random() < p.gamma_switch*0.2):
            return State.E_R
        if rng.random() < p.delta_decay_f:
            return State.S
        return s

    if s == State.E_R:
        if saw_r and (rng.random() < eff_share_r):
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
