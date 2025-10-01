import numpy as np
from .states import State, Params
from .grid import posting_neighbour_flags

def seed_initial(N, seeds_f0, seeds_r0, rng):
    """
    Create an N x N grid, fill with Susceptible,
    then randomly place fake and real posters.
    """
    state = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N*N, size=total, replace=False)
        state.flat[idx[:seeds_f0]] = State.I_F
        state.flat[idx[seeds_f0:]] = State.I_R
    return state

def step(state, p: Params, rng):
    """
    Advance the grid by one tick using minimal rules:
    - People see neighbours' posts with prob beta_see
    - Exposed share with prob beta_share_f/r
    - Posters last one tick, then step down to exposed
    """
    N = state.shape[0]
    next_state = state.copy()

    has_f, has_r = posting_neighbour_flags(state)
    sees_f = has_f & (rng.random((N, N)) < p.beta_see)
    sees_r = has_r & (rng.random((N, N)) < p.beta_see)

    # Masks for current states
    s_mask  = (state == State.S)
    ef_mask = (state == State.E_F)
    er_mask = (state == State.E_R)
    if_mask = (state == State.I_F)
    ir_mask = (state == State.I_R)

    # Susceptible -> Exposed if they saw
    next_state[s_mask & sees_f] = State.E_F
    next_state[s_mask & sees_r] = State.E_R

    # Exposed -> Posting with share probability
    next_state[ef_mask & (rng.random((N, N)) < p.beta_share_f)] = State.I_F
    next_state[er_mask & (rng.random((N, N)) < p.beta_share_r)] = State.I_R

    # Posters last one tick -> then exposed
    next_state[if_mask] = State.E_F
    next_state[ir_mask] = State.E_R

    return next_state

def simulate(p: Params):
    """
    Run T ticks and collect results.
    Returns dict with time series + final reach.
    """
    rng = np.random.default_rng(p.rng_seed)
    state = seed_initial(p.N, p.seeds_f0, p.seeds_r0, rng)

    I_f, I_r = [], []
    for _ in range(p.T):
        I_f.append(int((state == State.I_F).sum()))
        I_r.append(int((state == State.I_R).sum()))
        state = step(state, p, rng)

    return {"I_f": I_f, "I_r": I_r}
