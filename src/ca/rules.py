from __future__ import annotations
from typing import Tuple
import numpy as np
from .states import State, Params

def _coin(rng: np.random.Generator, p: float) -> bool:
    """Bernoulli(p) using the provided RNG."""
    # Clamp tiny numeric drift
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
    Transition one cell by one tick given:
      - current: current State enum value (int compatible)
      - saw_f/saw_r: whether at least one neighbouring fake/real poster is visible
      - params: simulation parameters
      - rng: numpy Generator for determinism
      - cooldown: remaining refractory ticks (applies when micro_refractory is True)

    Returns:
      (new_state, new_cooldown, share_f, share_r)
      - share_f/share_r are 1 if the agent newly posts fake/real in THIS tick, else 0.
    """
    s = State(current)
    share_f = 0
    share_r = 0

    # If refractory is enabled AND we are in cooldown, just tick down and keep state.
    if params.micro_refractory and cooldown > 0:
        return int(s), max(0, cooldown - 1), 0, 0

    # --- Helper lambdas for decays
    def decay_from_EF(cur_state: State) -> State:
        # Exposed-Fake can forget
        if _coin(rng, params.delta_decay_f):
            return State.S
        return cur_state

    def decay_from_ER(cur_state: State) -> State:
        # Exposed-Real can forget
        if _coin(rng, params.delta_decay_r):
            return State.S
        return cur_state

    def decay_from_IF() -> State:
        # Posters of Fake can drop to Exposed-Fake (not straight to S)
        if _coin(rng, params.delta_decay_f):
            return State.E_F
        return State.I_F

    def decay_from_IR() -> State:
        # Posters of Real can drop to Exposed-Real
        if _coin(rng, params.delta_decay_r):
            return State.E_R
        return State.I_R

    # --- Behaviour by current state
    if s == State.S:
        # Decide exposure if they "notice" neighbours.
        if (saw_f or saw_r) and _coin(rng, params.beta_see):
            if saw_f and saw_r:
                # Both visible: break tie neutrally
                s = State.E_F if _coin(rng, 0.5) else State.E_R
            elif saw_f:
                s = State.E_F
            else:
                s = State.E_R
        # From S there is no immediate posting; return as-is.
        return int(s), 0, 0, 0

    if s == State.E_F:
        # Base probability to share fake after exposure
        p_f = params.beta_share_f
        # If real is also visible, correction suppresses sharing fake
        if saw_r:
            p_f *= (1.0 - params.gamma_correction)
        if _coin(rng, p_f):
            s = State.I_F
            share_f = 1
            # Enter refractory if enabled
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        # Otherwise, apply decay while staying exposed
        s = decay_from_EF(s)
        return int(s), 0, share_f, share_r

    if s == State.E_R:
        # Probability to share real
        p_r = params.beta_share_r
        if _coin(rng, p_r):
            s = State.I_R
            share_r = 1
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        # Otherwise, apply decay while staying exposed
        s = decay_from_ER(s)
        return int(s), 0, share_f, share_r

    if s == State.I_F:
        # Option to switch to real when seeing real content
        if saw_r and _coin(rng, params.gamma_switch):
            s = State.I_R
            # Switching is becoming a real poster; count as a real share THIS tick
            share_r = 1
            cd = params.tau_post if params.micro_refractory else 0
            return int(s), cd, share_f, share_r
        # Otherwise apply decay while posting fake
        s = decay_from_IF()
        # If we decayed to E_F, cooldown resets; if still I_F and refractory on, lock again
        cd = params.tau_post if (params.micro_refractory and s in (State.I_F,)) else 0
        return int(s), cd, share_f, share_r

    if s == State.I_R:
        # (No real->fake switching by default; asymmetry intentional.)
        # Apply decay while posting real
        s = decay_from_IR()
        cd = params.tau_post if (params.micro_refractory and s in (State.I_R,)) else 0
        return int(s), cd, share_f, share_r

    # Fallback (shouldn't happen)
    return int(s), 0, share_f, share_r
