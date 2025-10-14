from __future__ import annotations
import numpy as np
from dataclasses import replace
from .states import State, Params
from .grid import posting_neighbour_flags, posting_neighbour_flags_local
from .rules import update_cell


# -------------------------------------------------------------------
#  Grid initialisation
# -------------------------------------------------------------------

def seed_initial(N: int, seeds_f0: int, seeds_r0: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create an N×N lattice where all agents start as Susceptible (S),
    except for the chosen number of initial fake and real posters.
    """
    grid = np.full((N, N), State.S, dtype=np.int8)
    total = seeds_f0 + seeds_r0
    if total > 0:
        idx = rng.choice(N * N, size=total, replace=False)
        grid.flat[idx[:seeds_f0]] = State.I_F
        grid.flat[idx[seeds_f0:total]] = State.I_R
    return grid


# -------------------------------------------------------------------
#  Local perception noise (misclassification)
# -------------------------------------------------------------------

def _apply_misclass_mask(rng: np.random.Generator, saw_f: np.ndarray, saw_r: np.ndarray, eta: float):
    """Randomly swap fake/real perception with probability `eta`."""
    if eta <= 0.0:
        return saw_f, saw_r
    mask = rng.random(saw_f.shape) < eta
    sf_new = np.where(mask, saw_r, saw_f)
    sr_new = np.where(mask, saw_f, saw_r)
    return sf_new, sr_new


# -------------------------------------------------------------------
#  Synchronous and asynchronous tick steps
# -------------------------------------------------------------------

def _tick_sync(state: np.ndarray,
               cooldown: np.ndarray | None,
               rng: np.random.Generator,
               p: Params,
               beta_see_eff: np.ndarray,
               beta_share_f_eff: np.ndarray,
               beta_share_r_eff: np.ndarray):
    """Advance all agents simultaneously using local effective parameters."""
    saw_f, saw_r = posting_neighbour_flags(state)
    if p.micro_misclass and p.eta_misclass > 0.0:
        saw_f, saw_r = _apply_misclass_mask(rng, saw_f, saw_r, p.eta_misclass)

    N = p.N
    next_state = state.copy()
    new_f = new_r = 0
    ever_fake = (saw_f | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r | (state == State.I_R) | (state == State.E_R))

    for i in range(N):
        for j in range(N):
            cd_in = int(cooldown[i, j]) if (p.micro_refractory and cooldown is not None) else 0

            p_loc = replace(
                p,
                beta_see=float(beta_see_eff[i, j]),
                beta_share_f=float(beta_share_f_eff[i, j]),
                beta_share_r=float(beta_share_r_eff[i, j]),
            )

            ns, cd_out, sf, sr = update_cell(
                int(state[i, j]),
                bool(saw_f[i, j]),
                bool(saw_r[i, j]),
                p_loc,
                rng,
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
                p: Params,
                beta_see_eff: np.ndarray,
                beta_share_f_eff: np.ndarray,
                beta_share_r_eff: np.ndarray):
    """Advance agents one at a time in random order (asynchronous update)."""
    N = p.N
    order = rng.permutation(N * N)
    new_f = new_r = 0

    saw_f0, saw_r0 = posting_neighbour_flags(state)
    if p.micro_misclass and p.eta_misclass > 0.0:
        saw_f0, saw_r0 = _apply_misclass_mask(rng, saw_f0, saw_r0, p.eta_misclass)

    ever_fake = (saw_f0 | (state == State.I_F) | (state == State.E_F))
    ever_real = (saw_r0 | (state == State.I_R) | (state == State.E_R))

    for idx in order:
        i, j = divmod(idx, N)
        sf_loc, sr_loc = posting_neighbour_flags_local(state, i, j)

        if p.micro_misclass and p.eta_misclass > 0.0 and rng.random() < p.eta_misclass:
            sf_loc, sr_loc = sr_loc, sf_loc

        p_loc = replace(
            p,
            beta_see=float(beta_see_eff[i, j]),
            beta_share_f=float(beta_share_f_eff[i, j]),
            beta_share_r=float(beta_share_r_eff[i, j]),
        )

        cd_in = int(cooldown[i, j]) if (p.micro_refractory and cooldown is not None) else 0
        ns, cd_out, sf, sr = update_cell(
            int(state[i, j]),
            bool(sf_loc),
            bool(sr_loc),
            p_loc,
            rng,
            cooldown=cd_in,
        )
        state[i, j] = ns
        if p.micro_refractory and cooldown is not None:
            cooldown[i, j] = cd_out
        new_f += int(sf)
        new_r += int(sr)

    return state, cooldown, new_f, new_r, ever_fake, ever_real


# -------------------------------------------------------------------
#  Simulation driver
# -------------------------------------------------------------------

def simulate(p: Params) -> dict:
    """
    Run the full simulation for T time steps using configuration `p`.
    Returns a dictionary containing time series and summary statistics.
    """
    rng = np.random.default_rng(p.rng_seed)
    seed_used = int(rng.integers(0, 2**32 - 1)) if p.rng_seed is None else p.rng_seed
    if p.rng_seed is None:
        rng = np.random.default_rng(seed_used)

    N, T = p.N, p.T
    state = seed_initial(N, p.seeds_f0, p.seeds_r0, rng)

    cooldown = np.zeros((N, N), dtype=np.int16) if p.micro_refractory else None

    # --- macro fields ---
    if p.macro_hetero and getattr(p, "hetero_sd", 0.0) > 0.0:
        hetero_field = np.clip(
            rng.lognormal(mean=0.0, sigma=float(p.hetero_sd), size=(N, N)).astype(np.float32),
            0.25, 4.0
        )
    else:
        hetero_field = np.ones((N, N), dtype=np.float32)

    if p.macro_spatial and float(getattr(p, "spatial_strength", 0.0)) > 0.0:
        yy, xx = np.mgrid[0:N, 0:N]
        cy, cx = (N - 1) / 2.0, (N - 1) / 2.0
        distance = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_d = np.sqrt(2) * ((N - 1) / 2.0)
        spatial_field = np.exp(-float(p.spatial_strength) * (distance / max_d)).astype(np.float32)
        spatial_field = np.clip(spatial_field, 0.25, 1.0)
    else:
        spatial_field = np.ones((N, N), dtype=np.float32)

    multiplier = np.clip(hetero_field * spatial_field, 0.1, 4.0)
    beta_see_eff = np.clip(float(p.beta_see) * spatial_field, 0.0, 1.0).astype(np.float32)
    beta_share_f_eff = np.clip(float(p.beta_share_f) * multiplier, 0.0, 1.0).astype(np.float32)
    beta_share_r_eff = np.clip(float(p.beta_share_r) * multiplier, 0.0, 1.0).astype(np.float32)

    # --- time series recording ---
    I_f, I_r, shares_f, shares_r = [], [], [], []
    ever_fake = np.zeros((N, N), dtype=bool)
    ever_real = np.zeros((N, N), dtype=bool)

    for _ in range(T):
        if p.update_scheme == "async" or p.micro_async:
            state, cooldown, new_f, new_r, ef, er = _tick_async(
                state, cooldown, rng, p,
                beta_see_eff, beta_share_f_eff, beta_share_r_eff
            )
        else:
            state, cooldown, new_f, new_r, ef, er = _tick_sync(
                state, cooldown, rng, p,
                beta_see_eff, beta_share_f_eff, beta_share_r_eff
            )

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
        "hetero_field": hetero_field.tolist(),
        "spatial_field": spatial_field.tolist(),
        "params": dict(p.__dict__),
    }
