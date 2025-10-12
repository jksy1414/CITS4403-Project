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

# ---------- refractory helpers ----------

def _decay_cooldown_in_place(cooldown: np.ndarray, mask_exempt: np.ndarray | None = None):
    """
    Decrement cooldown where >0. If mask_exempt is provided, those cells
    are excluded from decrement this tick (for 'just entered' posters).
    """
    if mask_exempt is None:
        np.subtract(cooldown, (cooldown > 0), out=cooldown, where=(cooldown > 0))
    else:
        dec_mask = (cooldown > 0) & (~mask_exempt)
        cooldown[dec_mask] -= 1

# ---------- misclassification helper (M3) ----------

def _apply_misclass(saw_f: bool, saw_r: bool, rng: np.random.Generator, p: Params) -> tuple[bool, bool]:
    """
    With probability eta_misclass, flip the perception fake<->real if exactly one of them is seen.
    If both or none are seen, leave unchanged (simple symmetric model).
    """
    if not getattr(p, "micro_misclass", False) or p.eta_misclass <= 0.0:
        return saw_f, saw_r

    if rng.random() < p.eta_misclass:
        if saw_f and not saw_r:
            return False, True
        if saw_r and not saw_f:
            return True, False
    return saw_f, saw_r

# ---------- sync tick ----------

def _tick_sync(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params):
    """
    Synchronous update:
      - compute saw_f/saw_r for whole grid
      - transition to next_state independently
      - if micro_refractory: posters with cooldown>0 are locked (no change)
      - set cooldown = tau_post for cells that just became posters
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

        if p.micro_refractory and s in (State.I_F, State.I_R) and cooldown[i, j] > 0:
            ns = s  # locked
        else:
            ns = update_cell(s, bool(saw_f[i, j]), bool(saw_r[i, j]), rng, p)

        next_state[i, j] = ns

        if s != State.I_F and ns == State.I_F:
            new_shares_f += 1
            entered_poster_mask[i, j] = True
        if s != State.I_R and ns == State.I_R:
            new_shares_r += 1
            entered_poster_mask[i, j] = True

        it.iternext()

    # cooldown bookkeeping
    _decay_cooldown_in_place(next_cooldown)
    if p.micro_refractory and p.tau_post > 0:
        next_cooldown[entered_poster_mask] = p.tau_post

    return next_state, next_cooldown, new_shares_f, new_shares_r

# ---------- async tick ----------

def _tick_async(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params):
    """
    Asynchronous update in random order, in-place.
    If micro_refractory: a poster with cooldown>0 is locked (no change).
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

        if p.micro_refractory and s in (State.I_F, State.I_R) and cooldown[i, j] > 0:
            continue  # locked poster; skip

        saw_f, saw_r = posting_neighbour_flags_local(state, i, j)
        ns = update_cell(s, saw_f, saw_r, rng, p)

        if s != State.I_F and ns == State.I_F:
            new_shares_f += 1
            entered_poster_mask[i, j] = True
        if s != State.I_R and ns == State.I_R:
            new_shares_r += 1
            entered_poster_mask[i, j] = True

        state[i, j] = ns

    # cooldown bookkeeping (don’t decrement those that just became posters)
    if p.micro_refractory:
        _decay_cooldown_in_place(cooldown, mask_exempt=entered_poster_mask)
        if p.tau_post > 0:
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

    # cooldown timers (harmless if micro_refractory=False)
    cooldown = np.zeros((N, N), dtype=np.int16)

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        # reach bookkeeping at tick boundary (comparable across schemes)
        snap = state.copy()
        ever_fake |= ((snap == State.I_F) | (snap == State.E_F))
        ever_real |= ((snap == State.I_R) | (snap == State.E_R))

        if p.update_scheme == "async":
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
