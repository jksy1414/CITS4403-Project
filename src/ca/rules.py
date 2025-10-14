from __future__ import annotations
from typing import Tuple
import numpy as np
from .states import Params, State


def _coin(rng: np.random.Generator, p: float) -> bool:
    """Bernoulli(p) using the provided RNG (with simple clamping for edge drift)."""
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
    Advance one agent by a single tick.

    Args:
        current: current State enum value (int).
        saw_f/saw_r: whether the local neighbourhood shows fake/real posters.
        params: simulation parameters.
        rng: numpy Generator for all randomness.
        cooldown: remaining refractory ticks (if micro_refractory is enabled).

    Returns:
        (new_state, new_cooldown, share_f, share_r)
        where share_* == 1 iff the agent starts posting that type THIS tick.
    """
    s = State(current)
    share_f = 0
    share_r = 0

    # Refractory: if enabled and still cooling down, tick it and keep state.
    if params.micro_refractory and cooldown > 0:
        return int(s), max(0, cooldown - 1), 0, 0

    # --- Small local helpers (pure style; identical semantics) ----------------
    def decay_from_EF(cur: State) -> State:
        # Exposed-Fake can forget to S
        return State.S if _coin(rng, params.delta_decay_f) else cur

    def decay_from_ER(cur: State) -> State:
        # Exposed-Real can forget to S
        return State.S if _coin(rng, params.delta_decay_r) else cur

    def decay_from_IF() -> State:
        # Fake poster can drop to Exposed-Fake (not straight to S)
        return State.E_F if _coin(rng, params.delta_decay_f) else State.I_F

    def decay_from_IR() -> State:
        # Real poster can drop to Exposed-Real
        return State.E_R if _coin(rng, params.delta_decay_r) else State.I_R

    # --- State machine --------------------------------------------------------
    if s == State.S:
        # Possibly become exposed if anything was noticed
        if (saw_f or saw_r) and _coin(rng, params.beta_see):
            if saw_f and saw_r:
                # Pick a side neutrally when both are visible
                s = State.E_F if _coin(rng, 0.5) else State.E_R
            elif saw_f:
                s = State.E_F
            else:
                s = State.E_R
        # No immediate posting from S
        return int(s), 0, 0, 0

    if s == State.E_F:
        # Base chance to start posting fake
        p_f = params.beta_share_f
        # Seeing real content tempers fake sharing
        if saw_r:
            p_f *= (1.0 - params.gamma_correction)
        if _coin(rng, p_f):
            s = State.I_F
            share_f = 1
            next_cd = params.tau_post if params.micro_refractory else 0
            return int(s), next_cd, share_f, 0
        # Otherwise stay exposed with possible decay to S
        s = decay_from_EF(s)
        return int(s), 0, share_f, 0

    if s == State.E_R:
        # Probability to start posting real
        if _coin(rng, params.beta_share_r):
            s = State.I_R
            share_r = 1
            next_cd = params.tau_post if params.micro_refractory else 0
            return int(s), next_cd, 0, share_r
        # Otherwise stay exposed with possible decay to S
        s = decay_from_ER(s)
        return int(s), 0, 0, share_r

    if s == State.I_F:
        # May switch to real if seeing real content (asymmetric by design)
        if saw_r and _coin(rng, params.gamma_switch):
            s = State.I_R
            share_r = 1  # switching counts as a real share this tick
            next_cd = params.tau_post if params.micro_refractory else 0
            return int(s), next_cd, 0, share_r

        # Otherwise apply attention decay while posting fake
        s = decay_from_IF()
        # If still posting fake and refractory is on, lock again
        next_cd = params.tau_post if (params.micro_refractory and s in (State.I_F,)) else 0
        return int(s), next_cd, 0, 0

    if s == State.I_R:
        # No default switch back to fake.
        s = decay_from_IR()
        next_cd = params.tau_post if (params.micro_refractory and s in (State.I_R,)) else 0
        return int(s), next_cd, 0, 0

    # Defensive default (shouldn't occur)
    return int(s), 0, share_f, share_r
