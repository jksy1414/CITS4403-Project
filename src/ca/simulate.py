# src/ca/simulate.py
from __future__ import annotations
import numpy as np
from .states import State, Params
from .grid import posting_neighbour_flags, posting_neighbour_flags_local
from .rules import update_cell

# ---------- Seeding ----------

def seed_initial(N: int, seeds_f0: int, seeds_r0: int, rng: np.random.Generator) -> np.ndarray:
    """Initialise an N×N grid with seeds_f0 fake posters and seeds_r0 real posters."""
    state = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N * N, size=total, replace=False)
        state.flat[idx[:seeds_f0]] = State.I_F
        state.flat[idx[seeds_f0:total]] = State.I_R
    return state


# ---------- Optional macro helpers (kept for future use) ----------

def _hetero_multipliers(N: int, rng: np.random.Generator, sd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two NxN fields (for fake, real) ~ lognormal noise with mean ~1.
    sd is the stdev of the underlying normal; we clamp to [0.3, 2.0] to avoid extremes.
    (Note: currently not wired into rules; kept for future M5 work.)
    """
    if sd <= 0:
        ones = np.ones((N, N), dtype=float)
        return ones, ones
    zf = rng.normal(0.0, sd, size=(N, N))
    zr = rng.normal(0.0, sd, size=(N, N))
    f_mult = np.clip(np.exp(zf), 0.3, 2.0)
    r_mult = np.clip(np.exp(zr), 0.3, 2.0)
    return f_mult, r_mult


def _spatial_field(N: int, strength: float, mode: str = "radial") -> np.ndarray:
    """
    Spatial weight W[i,j] in [0,1]. W=1 means no change, lower values damp sharing/seeing.
    We construct a base pattern G in [0,1], then mix: W = (1-strength) + strength*G.
    (Note: currently not wired into rules; kept for future M6 work.)
    """
    if strength <= 0.0:
        return np.ones((N, N), dtype=float)

    y = np.linspace(-1, 1, N)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    if mode == "x-gradient":
        G = (X + 1.0) / 2.0  # 0..1 left->right
    else:
        R2 = X**2 + Y**2     # radial bump: 1 at center, ~0 at corners
        G = np.clip(1.0 - R2, 0.0, 1.0)

    W = (1.0 - strength) + strength * G
    return np.clip(W, 0.0, 1.0)


# ---------- Core ticks ----------

def _apply_misclass_mask(rng: np.random.Generator, saw_f: np.ndarray, saw_r: np.ndarray, eta: float):
    """Flip saw_f/saw_r elementwise with prob eta (simple swap) to model misperception."""
    if eta <= 0.0:
        return saw_f, saw_r
    mask = rng.random(saw_f.shape) < eta
    sf = saw_f.copy()
    sr = saw_r.copy()
    saw_f = np.where(mask, sr, sf)
    saw_r = np.where(mask, sf, sr)
    return saw_f, saw_r


def _tick_sync(state: np.ndarray,
               cooldown: np.ndarray | None,
               rng: np.random.Generator,
               p: Params) -> tuple[np.ndarray, np.ndarray | None, int, int, np.ndarray, np.ndarray]:
    """Synchronous update: compute neighbour flags once, then update all cells from the snapshot."""
    saw_f, saw_r = posting_neighbour_flags(state)

    if p.micro_misclass and p.eta_misclass > 0.0:
        saw_f, saw_r = _apply_misclass_mask(rng, saw_f, saw_r, p.eta_misclass)

    # Reach is whether a cell has ever *seen or been* fake/real by this tick.
    ever_fake = (saw_f | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r | (state == State.I_R) | (state == State.E_R))

    N = p.N
    next_state = state.copy()
    new_f = 0
    new_r = 0

    for i in range(N):
        for j in range(N):
            cd_in = int(cooldown[i, j]) if (p.micro_refractory and cooldown is not None) else 0
            ns, cd_out, sf, sr = update_cell(
                int(state[i, j]),
                bool(saw_f[i, j]), bool(saw_r[i, j]),
                p, rng,
                cooldown=cd_in,
            )
            next_state[i, j] = ns
            if p.micro_refractory and cooldown is not None:
                cooldown[i, j] = cd_out
            new_f += int(sf)
            new_r += int(sr)

    return next_state, cooldown, new_f, new_r, ever_fake, ever_real


def _tick_async(state: np.ndarray,
                cooldown: np.ndarray | None,
                rng: np.random.Generator,
                p: Params) -> tuple[np.ndarray, np.ndarray | None, int, int, np.ndarray, np.ndarray]:
    """Asynchronous update: random visiting order; use local neighbour flags on the current, in-place state."""
    N = p.N
    order = rng.permutation(N * N)
    new_f = 0
    new_r = 0

    # We still accumulate reach from a snapshot at the beginning of the tick
    # (to remain comparable with sync). This is a design choice; OK for now.
    saw_f0, saw_r0 = posting_neighbour_flags(state)
    if p.micro_misclass and p.eta_misclass > 0.0:
        saw_f0, saw_r0 = _apply_misclass_mask(rng, saw_f0, saw_r0, p.eta_misclass)
    ever_fake = (saw_f0 | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r0 | (state == State.I_R) | (state == State.E_R))

    for idx in order:
        i, j = divmod(idx, N)

        # Local neighbour flags against the *current* state (in-place updating).
        sf_loc, sr_loc = posting_neighbour_flags_local(state, i, j)
        if p.micro_misclass and p.eta_misclass > 0.0:
            # Apply a per-cell coin for misclass in async path.
            if rng.random() < p.eta_misclass:
                sf_loc, sr_loc = sr_loc, sf_loc

        cd_in = int(cooldown[i, j]) if (p.micro_refractory and cooldown is not None) else 0
        ns, cd_out, sf, sr = update_cell(
            int(state[i, j]),
            bool(sf_loc), bool(sr_loc),
            p, rng,
            cooldown=cd_in,
        )
        state[i, j] = ns
        if p.micro_refractory and cooldown is not None:
            cooldown[i, j] = cd_out
        new_f += int(sf)
        new_r += int(sr)

    return state, cooldown, new_f, new_r, ever_fake, ever_real


# ---------- simulate ----------

def simulate(p: Params) -> dict:
    """Run the simulation for T steps and return time series and summary metrics."""
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    # Refractory cooldowns (micro)
    cooldown = None
    if p.micro_refractory:
        cooldown = np.zeros((N, N), dtype=np.int16)

    # Time series
    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        if p.update_scheme == "async" or p.micro_async:
            state, cooldown, new_f, new_r, ef, er = _tick_async(state, cooldown, rng, p)
        else:
            state, cooldown, new_f, new_r, ef, er = _tick_sync(state, cooldown, rng, p)

        ever_fake |= ef
        ever_real |= er
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
        "params": dict(p.__dict__),
    }
