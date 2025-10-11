import time
import numpy as np
from .states import State, Params
from .grid import posting_neighbour_flags


def seed_initial(N: int, seeds_f0: int, seeds_r0: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create an N x N grid, fill with Susceptible,
    then randomly place initial fake and real posters.
    """
    state = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N * N, size=total, replace=False)
        state.flat[idx[:seeds_f0]] = State.I_F
        state.flat[idx[seeds_f0:]] = State.I_R
    return state


def step(state: np.ndarray, p: Params, rng: np.random.Generator):
    """
    ONE TICK with minimal rules + correction/switching:
      - S who see F/R -> E_F / E_R
      - E_F / E_R may share -> I_F / I_R (we record these as share events)
      - I_F may switch to I_R (gamma_switch)
      - Posters last one tick -> drop to E_* (if not switched)
    Returns:
      next_state, share_f_mask, share_r_mask, switch_mask
    """
    N = state.shape[0]
    next_state = state.copy()

    # who has neighbours posting fake / real?
    has_f, has_r = posting_neighbour_flags(state)

    # who actually notices (sees) those neighbours?
    sees_f = has_f & (rng.random((N, N)) < p.beta_see)
    sees_r = has_r & (rng.random((N, N)) < p.beta_see)

    # current-state masks
    s_mask  = (state == State.S)
    ef_mask = (state == State.E_F)
    er_mask = (state == State.E_R)
    if_mask = (state == State.I_F)
    ir_mask = (state == State.I_R)

    # S -> E if they saw F or R
    next_state[s_mask & sees_f] = State.E_F
    next_state[s_mask & sees_r] = State.E_R

    # --- correction: if exposed to fake and also see real, reduce fake share prob
    # base sharing probabilities
    share_f_prob = p.beta_share_f
    share_r_prob = p.beta_share_r

    # reduce fake sharing for those exposed-to-fake who also saw real this tick
    ef_and_saw_real = ef_mask & sees_r
    # apply reduction only to those cells; everyone else keeps base prob
    # draw a fresh matrix of uniforms for E_F
    u_f = rng.random((N, N))
    share_f_mask = ef_mask & (u_f < share_f_prob)
    if p.gamma_correction > 0.0:
        # re-evaluate for corrected group with reduced probability
        reduced_prob = share_f_prob * (1.0 - p.gamma_correction)
        # remove previously marked shares in corrected group, then redraw for those group cells
        share_f_mask[ef_and_saw_real] = (u_f[ef_and_saw_real] < reduced_prob)

    # E_R share to I_R
    share_r_mask = er_mask & (rng.random((N, N)) < share_r_prob)

    # apply new posters
    next_state[share_f_mask] = State.I_F
    next_state[share_r_mask] = State.I_R

    # fake posters may switch to real
    switch_mask = if_mask & (rng.random((N, N)) < p.gamma_switch)
    next_state[switch_mask] = State.I_R  # switch instantly this tick

    # posters last one tick -> drop to exposed (if not switched)
    next_state[if_mask & ~switch_mask] = State.E_F
    next_state[ir_mask] = State.E_R

    return next_state, share_f_mask, share_r_mask, switch_mask


def simulate(p: Params) -> dict:
    """
    Run T ticks and collect results.

    Randomness:
      - If p.rng_seed is None -> use a time-based seed each run
      - Else use the provided fixed seed (reproducible)

    Returns a dict with:
      I_f, I_r               : per-tick number of fake/real posters (snapshot)
      shares_f, shares_r     : per-tick number of new shares (flow)
      switches               : per-tick fake->real switches
      cum_shares_f/_r        : totals over the run
      reach_fake/_real       : fraction of unique people ever exposed to fake/real
      seed_used              : the RNG seed used for this run
    """
    # choose seed
    if p.rng_seed is None:
        # time-based 32-bit seed (different each run)
        seed_used = int(time.time_ns() & 0xFFFFFFFF)
    else:
        seed_used = int(p.rng_seed)

    rng = np.random.default_rng(seed_used)

    # initial grid & seeds
    state = seed_initial(p.N, p.seeds_f0, p.seeds_r0, rng)

    # time series
    I_f, I_r = [], []
    shares_f, shares_r = [], []
    switches = []

    # reach tracking (ever seen F/R)
    ever_fake = np.zeros((p.N, p.N), dtype=bool)
    ever_real = np.zeros((p.N, p.N), dtype=bool)

    for _ in range(p.T):
        # record current posters (snapshot)
        I_f.append(int((state == State.I_F).sum()))
        I_r.append(int((state == State.I_R).sum()))

        # advance one tick; capture new shares and switches (flow)
        state, sf_mask, sr_mask, sw_mask = step(state, p, rng)
        shares_f.append(int(np.count_nonzero(sf_mask)))
        shares_r.append(int(np.count_nonzero(sr_mask)))
        switches.append(int(np.count_nonzero(sw_mask)))

        # update reach after state transition
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
        "reach_real": float(ever_real.mean()),
        "seed_used": seed_used,
    }
