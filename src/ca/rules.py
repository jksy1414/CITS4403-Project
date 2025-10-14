from __future__ import annotations
import numpy as np
from typing import Tuple
from .states import State, Params


def _coin(rng: np.random.Generator, p: float) -> bool:
    """Return True with probability p (Bernoulli trial)."""
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return rng.random() < p


def update_cell(
    current: int,
    saw_f: bool,
    saw_r: bool,
    params: Params,
    rng: np.random.Generator,
    cooldown: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Transition one cell (agent) forward by a single timestep.

    Inputs:
        current   – current State value (int or enum)
        saw_f/r   – booleans: whether fake/real posters are visible nearby
        params    – full simulation configuration
        rng       – NumPy random generator (for deterministic runs)
        cooldown  – remaining refractory ticks, if enabled

    Returns:
        (new_state, new_cooldown, share_f, share_r)
        where share_f/share_r = 1 if the cell newly posts fake/real this tick.
    """
    s = State(current)
    share_f = 0
    share_r = 0

    # --- Handle refractory lock ---
    if params.micro_refractory and cooldown > 0:
        return int(s), max(0, cooldown - 1), 0, 0

    # --- Local decay helper functions ---
    def decay_from_EF(st: State) -> State:
        return State.S if _coin(rng, params.delta_decay_f) else st

    def decay_from_ER(st: State) -> State:
        return State.S if _coin(rng, params.delta_decay_r) else st

    def decay_from_IF() -> State:
        return State.E_F if _coin(rng, params.delta_decay_f) else State.I_F

    def decay_from_IR() -> State:
        return State.E_R if _coin(rng, params.delta_decay_r) else State.I_R

    # --- Behavioural transitions ---
    if s == State.S:
        # Susceptible → exposed (fake/real)
        if (saw_f or saw_r) and _coin(rng, params.beta_see):
            if saw_f and saw_r:
                s = State.E_F if _coin(rng, 0.5) else State.E_R
            elif saw_f:
                s = State.E_F
            else:
                s = State.E_R
        return int(s), 0, 0, 0

    if s == State.E_F:
        # Exposed to fake — may start posting
        p_f = params.beta_share_f
        if saw_r:
            # If also seeing real, correction reduces fake spread
            p_f *= (1.0 - params.gamma_correction)
        if _coin(rng, p_f):
            s = State.I_F
            share_f = 1
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        s = decay_from_EF(s)
        return int(s), 0, share_f, share_r

    if s == State.E_R:
        # Exposed to real — may start posting
        p_r = params.beta_share_r
        if _coin(rng, p_r):
            s = State.I_R
            share_r = 1
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        s = decay_from_ER(s)
        return int(s), 0, share_f, share_r

    if s == State.I_F:
        # Active fake poster — can switch or decay
        if saw_r and _coin(rng, params.gamma_switch):
            # Fake → Real switching (corrected belief)
            s = State.I_R
            share_r = 1
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        s = decay_from_IF()
        cd = params.tau_post if params.micro_refractory and s == State.I_F else 0
        return int(s), cd, share_f, share_r

    if s == State.I_R:
        # Active real poster — decays normally
        s = decay_from_IR()
        cd = params.tau_post if params.micro_refractory and s == State.I_R else 0
        return int(s), cd, share_f, share_r

    # Default: should never happen
    return int(s), 0, share_f, share_r
