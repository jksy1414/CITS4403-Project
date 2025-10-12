from __future__ import annotations
import numpy as np
from .states import State, Params
from .grid import posting_neighbour_flags, posting_neighbour_flags_local
from .rules import update_cell

def seed_initial(N: int, seeds_f0: int, seeds_r0: int, rng: np.random.Generator) -> np.ndarray:
    state = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N * N, size=total, replace=False)
        state.flat[idx[:seeds_f0]] = State.I_F
        state.flat[idx[seeds_f0:total]] = State.I_R
    return state

# ---------- optional refractory helpers (safe even if you don't enable it) ----------

def _decay_cooldown_in_place(cooldown: np.ndarray, mask_exempt: np.ndarray | None = None):
    """Decrement cooldown>0, optionally skipping cells in mask_exempt."""
    if mask_exempt is None:
        np.subtract(cooldown, (cooldown > 0), out=cooldown, where=(cooldown > 0))
    else:
        dec_mask = (cooldown > 0) & (~mask_exempt)
        cooldown[dec_mask] -= 1

# ---------- sync tick ----------

def _tick_sync(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params):
    """
    Synchronous tick:
      1) compute neighbour flags for the whole grid
      2) compute next_state independently (no in-place coupling)
      3) handle refractory cooldowns if enabled
    Returns: next_state, next_cooldown, new_shares_f, new_shares_r
    """
    saw_f, saw_r = posting_neighbour_flags(state)
    next_state = state.copy()
    next_cooldown = cooldown.copy()
    new_shares_f = 0
    new_shares_r = 0

    N = state.shape[0]
    entered_poster_mask = np.zeros((N, N), dtype=bool)

    it = np.nditer(state, flags=['multi_index'])
    while not it.finished:
        i, j = it.multi_index
        s = int(it[0])

        # refractory lock for posters (only if enabled)
        if getattr(p, "micro_refractory", False) and s in (State.I_F, State.I_R) and cooldown[i, j] > 0:
            ns = s
        else:
            ns = update_cell(s, bool(saw_f[i, j]), bool(saw_r[i, j]), rng, p)

        next_state[i, j] = ns

        # count new posters
        if s != State.I_F and ns == State.I_F:
            new_shares_f += 1
            entered_poster_mask[i, j] = True
        if s != State.I_R and ns == State.I_R:
            new_shares_r += 1
            entered_poster_mask[i, j] = True

        it.iternext()

    # cooldown bookkeeping
    _decay_cooldown_in_place(next_cooldown)
    if getattr(p, "micro_refractory", False) and getattr(p, "tau_post", 0) > 0:
        next_cooldown[entered_poster_mask] = p.tau_post

    return next_state, next_cooldown, new_shares_f, new_shares_r

# ---------- async tick (NEW) ----------

def _tick_async(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params):
    """
    Asynchronous tick: update cells in random order *in-place*.
    Each decision reads neighbours from the current (partially updated) state.
    Refractory (if enabled): posters with cooldown>0 are locked.
    Returns: state, cooldown, new_shares_f, new_shares_r
    """
    N = state.shape[0]
    order = rng.permutation(N * N)
    new_shares_f = 0
    new_shares_r = 0

    entered_poster_mask = np.zeros((N, N), dtype=bool)

    for flat in order:
        i = flat // N
        j = flat % N
        s = int(state[i, j])

        # refractory lock (optional)
        if getattr(p, "micro_refractory", False) and s in (State.I_F, State.I_R) and cooldown[i, j] > 0:
            continue  # keep same posting state

        saw_f, saw_r = posting_neighbour_flags_local(state, i, j)
        ns = update_cell(s, saw_f, saw_r, rng, p)

        if s != State.I_F and ns == State.I_F:
            new_shares_f += 1
            entered_poster_mask[i, j] = True
        if s != State.I_R and ns == State.I_R:
            new_shares_r += 1
            entered_poster_mask[i, j] = True

        state[i, j] = ns

    # cooldown bookkeeping (avoid decrementing the ones that *just* became posters)
    if getattr(p, "micro_refractory", False):
        _decay_cooldown_in_place(cooldown, mask_exempt=entered_poster_mask)
        if getattr(p, "tau_post", 0) > 0:
            cooldown[entered_poster_mask] = p.tau_post
    else:
        _decay_cooldown_in_place(cooldown)

    return state, cooldown, new_shares_f, new_shares_r

# ---------- simulate ----------

def simulate(p: Params) -> dict:
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    # cooldown array always present (harmless if micro_refractory=False)
    cooldown = np.zeros((N, N), dtype=np.int16)

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        # reach snapshot at tick boundary (comparable across schemes)
        snap = state.copy()
        ever_fake |= ((snap == State.I_F) | (snap == State.E_F))
        ever_real |= ((snap == State.I_R) | (snap == State.E_R))

        if getattr(p, "update_scheme", "sync") == "async":
            state, cooldown, new_f, new_r = _tick_async(state, cooldown, rng, p)
        else:
            state, cooldown, new_f, new_r = _tick_sync(state, cooldown, rng, p)

        I_f.append(int(np.sum(state == State.I_F)))
        I_r.append(int(np.sum(state == State.I_R)))
        shares_f.append(new_f)
        shares_r.append(new_r)

    return {
        "N": N, "T": T, "seed_used": seed_used,
        "I_f": I_f, "I_r": I_r,
        "shares_f": shares_f, "shares_r": shares_r,
        "reach_fake": float(np.mean(ever_fake)),
        "reach_real": float(np.mean(ever_real)),
        "params": p.__dict__,
    }
