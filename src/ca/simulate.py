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

# ---------- Macro helpers ----------

def _hetero_multipliers(N: int, rng: np.random.Generator, sd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two NxN fields (for fake, real) ~ lognormal noise with mean ~1.
    sd is the stdev of the underlying normal; we clamp to [0.3, 2.0] to avoid extremes.
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
    - 'radial': center high, edges low
    - 'x-gradient': left low, right high
    """
    if strength <= 0.0:
        return np.ones((N, N), dtype=float)

    y = np.linspace(-1, 1, N)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    if mode == "x-gradient":
        G = (X + 1.0) / 2.0  # 0..1 left->right
    else:
        # radial bump: 1 at center, ~0 at corners
        R2 = X**2 + Y**2
        G = np.clip(1.0 - R2, 0.0, 1.0)

    W = (1.0 - strength) + strength * G
    # Clamp numerically safe
    return np.clip(W, 0.0, 1.0)

# ---------- Core ticks ----------

def _tick_sync(state, cooldown, rng, p: Params, f_mult, r_mult, w_field):
    saw_f, saw_r = posting_neighbour_flags(state)

    # misclassification view (prob η): flip saw_f/saw_r with small chance
    if p.micro_misclass and p.eta_misclass > 0.0:
        mask = rng.random(state.shape) < p.eta_misclass
        # three-way logic: we handle flips elementwise
        # create copies
        sf = saw_f.copy()
        sr = saw_r.copy()
        # where mask and both True/False -> leave as is or random swap
        # simpler: on mask, swap booleans
        saw_f = np.where(mask, sr, sf)
        saw_r = np.where(mask, sf, sr)

    # record REACH before updating
    ever_fake = (saw_f | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r | (state == State.I_R) | (state == State.E_R))

    N = p.N
    next_state = state.copy()
    new_f = 0
    new_r = 0

    for i in range(N):
        for j in range(N):
            # effective share probs at (i,j)
            sf = float(np.clip(p.beta_share_f * f_mult[i, j] * w_field[i, j], 0.0, 1.0))
            sr = float(np.clip(p.beta_share_r * r_mult[i, j] * w_field[i, j], 0.0, 1.0))

            # refractory: if cooling > 0, skip state changes (except decay down from I_* handled in rules)
            if p.micro_refractory and cooldown is not None and cooldown[i, j] > 0:
                # still allow decay while posting? We’ll respect cooldown by freezing transitions away from exposed,
                # but keep rules unified—so we pass through update_cell but afterwards we enforce no change unless decay.
                pass

            ns = update_cell(int(state[i, j]),
                             bool(saw_f[i, j]), bool(saw_r[i, j]),
                             rng, p, sf, sr)

            # refractory bookkeeping: if started posting now, set cooldown
            if p.micro_refractory and cooldown is not None:
                if state[i, j] != State.I_F and ns == State.I_F:
                    cooldown[i, j] = p.tau_post
                elif state[i, j] != State.I_R and ns == State.I_R:
                    cooldown[i, j] = p.tau_post
                else:
                    # tick down if >0
                    if cooldown[i, j] > 0:
                        cooldown[i, j] -= 1

            next_state[i, j] = ns
            if state[i, j] != State.I_F and ns == State.I_F: new_f += 1
            if state[i, j] != State.I_R and ns == State.I_R: new_r += 1

    return next_state, cooldown, new_f, new_r, ever_fake, ever_real

def _tick_async(state, cooldown, rng, p: Params, f_mult, r_mult, w_field):
    saw_f, saw_r = posting_neighbour_flags(state)

    if p.micro_misclass and p.eta_misclass > 0.0:
        mask = rng.random(state.shape) < p.eta_misclass
        sf = saw_f.copy(); sr = saw_r.copy()
        saw_f = np.where(mask, sr, sf)
        saw_r = np.where(mask, sf, sr)

    N = p.N
    order = rng.permutation(N * N)
    new_f = 0
    new_r = 0
    ever_fake = (saw_f | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r | (state == State.I_R) | (state == State.E_R))

    for idx in order:
        i, j = divmod(idx, N)
        sf = float(np.clip(p.beta_share_f * f_mult[i, j] * w_field[i, j], 0.0, 1.0))
        sr = float(np.clip(p.beta_share_r * r_mult[i, j] * w_field[i, j], 0.0, 1.0))

        ns = update_cell(int(state[i, j]),
                         bool(saw_f[i, j]), bool(saw_r[i, j]),
                         rng, p, sf, sr)

        if p.micro_refractory and cooldown is not None:
            if state[i, j] != State.I_F and ns == State.I_F:
                cooldown[i, j] = p.tau_post
            elif state[i, j] != State.I_R and ns == State.I_R:
                cooldown[i, j] = p.tau_post
            else:
                if cooldown[i, j] > 0:
                    cooldown[i, j] -= 1

        if state[i, j] != State.I_F and ns == State.I_F: new_f += 1
        if state[i, j] != State.I_R and ns == State.I_R: new_r += 1
        state[i, j] = ns  # async: write-through

    return state, cooldown, new_f, new_r, ever_fake, ever_real

# ---------- simulate ----------

def simulate(p: Params) -> dict:
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    # Macro fields (always defined → avoids NameError)
    f_mult, r_mult = (np.ones((N, N), dtype=float), np.ones((N, N), dtype=float))
    if p.macro_hetero:
        f_mult, r_mult = _hetero_multipliers(N, rng, p.hetero_sd)

    w_field = np.ones((N, N), dtype=float)
    if p.macro_spatial:
        w_field = _spatial_field(N, p.spatial_strength, p.spatial_mode)

    # Refractory cooldowns (micro M2)
    cooldown = None
    if p.micro_refractory:
        cooldown = np.zeros((N, N), dtype=np.int16)

    I_f, I_r = [], []
    shares_f, shares_r = [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        if p.update_scheme == "async" or p.micro_async:
            state, cooldown, new_f, new_r, ef, er = _tick_async(state, cooldown, rng, p, f_mult, r_mult, w_field)
        else:
            state, cooldown, new_f, new_r, ef, er = _tick_sync(state, cooldown, rng, p, f_mult, r_mult, w_field)

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
        "params": p.__dict__,
    }
