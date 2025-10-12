# src/ca/simulate.py
from __future__ import annotations
import numpy as np
from typing import Tuple
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

def _build_hetero_multipliers(N: int, rng: np.random.Generator, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (f_mult, r_mult) multiplicative maps for fake/real share probabilities.
    If heterogeneity is OFF, returns ones.
    With heterogeneity ON, draw lognormal factors with sd=hetero_sd (approx).
    """
    if getattr(p, "macro_hetero", False):
        sd = float(getattr(p, "hetero_sd", 0.20))
        # Use lognormal around 1.0 (median=1), sd controlled; clamp to a sane range
        mu = 0.0  # median ~ 1.0
        sigma = sd
        f_mult = rng.lognormal(mean=mu, sigma=sigma, size=(N, N)).astype(np.float32)
        r_mult = rng.lognormal(mean=mu, sigma=sigma, size=(N, N)).astype(np.float32)
        # Optional clamp to reduce outliers
        f_mult = np.clip(f_mult, 0.25, 4.0)
        r_mult = np.clip(r_mult, 0.25, 4.0)
        return f_mult, r_mult
    else:
        ones = np.ones((N, N), dtype=np.float32)
        return ones, ones

def _tick_sync(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params,
               f_mult: np.ndarray, r_mult: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    N = state.shape[0]
    saw_f, saw_r = posting_neighbour_flags(state)

    next_state = state.copy()
    new_shares_f = 0
    new_shares_r = 0

    for i in range(N):
        for j in range(N):
            s = int(state[i, j])

            # refractory handling (if enabled)
            if getattr(p, "micro_refractory", False) and cooldown[i, j] > 0:
                # only decay cooldown; state locked (no change)
                cooldown[i, j] = max(0, cooldown[i, j] - 1)
                continue

            # hetero-adjusted probs (then clamped in update_cell too)
            sf = float(np.clip(p.beta_share_f * f_mult[i, j], 0.0, 1.0))
            sr = float(np.clip(p.beta_share_r * r_mult[i, j], 0.0, 1.0))

            ns = update_cell(
                s,
                bool(saw_f[i, j]),
                bool(saw_r[i, j]),
                rng,
                p,
                share_f_prob=sf,
                share_r_prob=sr,
            )

            # set cooldown if newly started posting (when refractory active)
            if getattr(p, "micro_refractory", False):
                if s != State.I_F and ns == State.I_F:
                    cooldown[i, j] = getattr(p, "tau_post", 3)
                elif s != State.I_R and ns == State.I_R:
                    cooldown[i, j] = getattr(p, "tau_post", 3)

            # count new posts
            if s != State.I_F and ns == State.I_F: new_shares_f += 1
            if s != State.I_R and ns == State.I_R: new_shares_r += 1

            next_state[i, j] = ns

    return next_state, cooldown, new_shares_f, new_shares_r

def _tick_async(state: np.ndarray, cooldown: np.ndarray, rng: np.random.Generator, p: Params,
                f_mult: np.ndarray, r_mult: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    N = state.shape[0]
    # draw a fresh random order each step
    order = rng.permutation(N * N)
    new_shares_f = 0
    new_shares_r = 0

    for idx in order:
        i = idx // N
        j = idx % N
        s = int(state[i, j])

        # refractory handling (if enabled)
        if getattr(p, "micro_refractory", False) and cooldown[i, j] > 0:
            cooldown[i, j] = max(0, cooldown[i, j] - 1)
            continue

        # neighborhood from current (asynchronously updated) state
        saw_f, saw_r = posting_neighbour_flags(state)
        sf = float(np.clip(p.beta_share_f * f_mult[i, j], 0.0, 1.0))
        sr = float(np.clip(p.beta_share_r * r_mult[i, j], 0.0, 1.0))

        ns = update_cell(
            s,
            bool(saw_f[i, j]),
            bool(saw_r[i, j]),
            rng,
            p,
            share_f_prob=sf,
            share_r_prob=sr,
        )

        if getattr(p, "micro_refractory", False):
            if s != State.I_F and ns == State.I_F:
                cooldown[i, j] = getattr(p, "tau_post", 3)
            elif s != State.I_R and ns == State.I_R:
                cooldown[i, j] = getattr(p, "tau_post", 3)

        if s != State.I_F and ns == State.I_F: new_shares_f += 1
        if s != State.I_R and ns == State.I_R: new_shares_r += 1

        state[i, j] = ns  # async: write back immediately

    return state, cooldown, new_shares_f, new_shares_r

def simulate(p: Params) -> dict:
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    # heterogeneity multipliers (always defined)
    f_mult, r_mult = _build_hetero_multipliers(N, rng, p)

    # cooldown (for refractory; still created even if feature off)
    cooldown = np.zeros((N, N), dtype=np.int16)

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        # snapshot of posters for reach bookkeeping
        sawF, sawR = posting_neighbour_flags(state)
        ever_fake |= (sawF | (state == State.I_F) | (state == State.E_F))
        ever_real |= (sawR | (state == State.I_R) | (state == State.E_R))

        if getattr(p, "update_scheme", "sync") == "async" or getattr(p, "micro_async", False):
            state, cooldown, new_f, new_r = _tick_async(state, cooldown, rng, p, f_mult, r_mult)
        else:
            state, cooldown, new_f, new_r = _tick_sync(state, cooldown, rng, p, f_mult, r_mult)

        I_f.append(int(np.sum(state == State.I_F)))
        I_r.append(int(np.sum(state == State.I_R)))
        shares_f.append(int(new_f))
        shares_r.append(int(new_r))

    reach_fake = float(np.mean(ever_fake))
    reach_real = float(np.mean(ever_real))

    return {
        "N": N, "T": T, "seed_used": seed_used,
        "I_f": I_f, "I_r": I_r,
        "shares_f": shares_f, "shares_r": shares_r,
        "reach_fake": reach_fake, "reach_real": reach_real,
        "params": p.__dict__,
    }
