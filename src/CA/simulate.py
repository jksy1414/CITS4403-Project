# src/ca/simulate.py
import numpy as np
from .states import State, Params
from .grid import posting_neighbour_flags
from .rules import update_cell

def seed_initial(N: int, seeds_f0: int, seeds_r0: int, rng: np.random.Generator) -> np.ndarray:
    state = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N * N, size=total, replace=False)
        state.flat[idx[:seeds_f0]] = State.I_F
        state.flat[idx[seeds_f0:total]] = State.I_R
    return state

def simulate(p: Params) -> dict:
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        saw_f, saw_r = posting_neighbour_flags(state)

        # reach bookkeeping BEFORE transitions
        ever_fake |= (saw_f | (state == State.I_F) | (state == State.E_F))
        ever_real |= (saw_r | (state == State.I_R) | (state == State.E_R))

        next_state = state.copy()
        new_shares_f = 0
        new_shares_r = 0

        it = np.nditer(state, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            s = int(it[0])
            ns = update_cell(s, bool(saw_f[i, j]), bool(saw_r[i, j]), rng, p)
            next_state[i, j] = ns
            if s != State.I_F and ns == State.I_F: new_shares_f += 1
            if s != State.I_R and ns == State.I_R: new_shares_r += 1
            it.iternext()

        state = next_state
        I_f.append(int(np.sum(state == State.I_F)))
        I_r.append(int(np.sum(state == State.I_R)))
        shares_f.append(new_shares_f)
        shares_r.append(new_shares_r)

    return {
        "N": N, "T": T, "seed_used": seed_used,
        "I_f": I_f, "I_r": I_r,
        "shares_f": shares_f, "shares_r": shares_r,
        "reach_fake": float(np.mean(ever_fake)),
        "reach_real": float(np.mean(ever_real)),
        "params": p.__dict__,
    }
