import numpy as np
from .states import State, Params

def update_cell(curr: int, saw_f: bool, saw_r: bool, rng: np.random.Generator, p: Params) -> int:
    s = curr
    # sharing probabilities (with correction)
    share_f_prob = p.beta_share_f
    share_r_prob = p.beta_share_r
    if saw_r:
        # seeing real lowers chance to share fake
        share_f_prob *= (1.0 - p.gamma_correction)

    # Transitions
    if s == State.S:
        if saw_f and not saw_r:
            return State.E_F
        if saw_r and not saw_f:
            return State.E_R
        if saw_f and saw_r:
            # tie-break: lean towards real (or random, but bias real slightly)
            return State.E_R if rng.random() < 0.6 else State.E_F
        return s

    if s == State.E_F:
        # may start posting fake
        if saw_f and (rng.random() < share_f_prob):
            return State.I_F
        # may convert to E_R if seeing real strongly
        if saw_r and (rng.random() < p.gamma_switch*0.2):
            return State.E_R
        # forgetting
        if rng.random() < p.delta_decay_f:
            return State.S
        return s

    if s == State.E_R:
        # may start posting real
        if saw_r and (rng.random() < share_r_prob):
            return State.I_R
        # forgetting
        if rng.random() < p.delta_decay_r:
            return State.S
        return s

    if s == State.I_F:
        # either stop posting (decay) or switch to I_R
        if saw_r and (rng.random() < p.gamma_switch):
            return State.I_R
        if rng.random() < p.delta_decay_f:
            return State.E_F  # drop to exposed (still remembers)
        return s

    if s == State.I_R:
        if rng.random() < p.delta_decay_r:
            return State.E_R
        return s

    return s