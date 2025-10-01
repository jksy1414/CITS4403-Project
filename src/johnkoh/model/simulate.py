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

    # --- Correction behaviour ---
    # Exposed to Fake -> less likely to share if also saw Real
    corrected_fake = ef_mask & sees_r
    share_f_prob = p.beta_share_f * (1 - p.gamma_correction)
    share_r_prob = p.beta_share_r

    # Shares this tick
    share_f_mask = ef_mask & (rng.random((N, N)) < share_f_prob)
    share_r_mask = er_mask & (rng.random((N, N)) < share_r_prob)

    next_state[share_f_mask] = State.I_F
    next_state[share_r_mask] = State.I_R

    # Fake posters may switch to Real
    switch_mask = if_mask & (rng.random((N, N)) < p.gamma_switch)
    next_state[switch_mask] = State.I_R

    # Posters last one tick -> drop to exposed (if not switched)
    next_state[if_mask & ~switch_mask] = State.E_F
    next_state[ir_mask] = State.E_R

    return next_state, share_f_mask, share_r_mask, switch_mask

def simulate(p: Params):
    """
    Run T ticks and collect results.
    Returns dict with time series + final reach.
    """
    rng = np.random.default_rng(p.rng_seed)
    state = np.full((p.N, p.N), State.S, dtype=np.int8)
    total = p.seeds_f0 + p.seeds_r0
    if total > 0:
        idx = rng.choice(p.N * p.N, size=total, replace=False)
        state.flat[idx[:p.seeds_f0]] = State.I_F
        state.flat[idx[p.seeds_f0:]] = State.I_R

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    switches = []

    # Track reach
    ever_fake = np.zeros((p.N, p.N), dtype=bool)
    ever_real = np.zeros((p.N, p.N), dtype=bool)

    for _ in range(p.T):
        # record current posters
        I_f.append(int((state == State.I_F).sum()))
        I_r.append(int((state == State.I_R).sum()))

        # advance one step
        state, sf_mask, sr_mask, sw_mask = step(state, p, rng)
        shares_f.append(int(np.count_nonzero(sf_mask)))
        shares_r.append(int(np.count_nonzero(sr_mask)))
        switches.append(int(np.count_nonzero(sw_mask)))

        # update reach
        ever_fake |= (state == State.E_F) | (state == State.I_F)
        ever_real |= (state == State.E_R) | (state == State.I_R)

    return {
        "I_f": I_f,
        "I_r": I_r,
        "shares_f": shares_f,
        "shares_r": shares_r,
        "switches": switches,
        "cum_shares_f": int(np.sum(shares_f)),
        "cum_shares_r": int(np.sum(shares_r)),
        "reach_fake": float(ever_fake.mean()),
        "reach_real": float(ever_real.mean())
    }
