from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

# Basic cell states (kept small/int for speed later)
class State(IntEnum):
    S   = 0  # Susceptible: hasn't seen anything
    E_F = 1  # Exposed to Fake (seen, not posting)
    I_F = 2  # Posting Fake
    E_R = 3  # Exposed to Real
    I_R = 4  # Posting Real

@dataclass
class Params:
    # World / timing
    N: int = 50                   # grid N x N (population size for CA)
    T: int = 60                   # number of time steps

    # Behaviour probabilities
    beta_see: float = 0.35
    beta_share_f: float = 0.50
    beta_share_r: float = 0.40

    # Initial posters conditions
    seeds_f0: int = 20
    seeds_r0: int = 10

    # Correction / switch mechanics
    gamma_correction: float = 0.70  # seeing real lowers chance to share fake
    gamma_switch: float = 0.30      # posters of fake may switch to real

    # Natural decay of attention (forgetting)
    delta_decay_f: float = 0.12     # fake forgets faster
    delta_decay_r: float = 0.06     # real forgets slower

    # Randomness
    rng_seed: Optional[int] = None  # None => random seed each run

    # Update scheme
    update_scheme: str = "sync"     # "sync" or "async"

    # Micro features
    micro_async: bool = False          # will mirror update_scheme
    micro_refractory: bool = False     # cooldown τ
    micro_misclass: bool = False       # enable misclassification

    # Macro features
    macro_hetero: bool = False
    macro_spatial: bool = False

    # Refractory configuration
    tau_post: int = 3                  # number of ticks a poster is 'locked'

    # Misclassification configuration
    eta_misclass: float = 0.02         # probability to misread fake<->real

    # Heterogeneity / spatial configuration
    hetero_sd: float = 0.20            # std dev for multiplicative noise
    spatial_strength: float = 0.35     # 0=no effect, 1=strong spatial modulation
    spatial_mode: str = "radial"       # "radial" or "x-gradient"

    def __post_init__(self):
        # Basic sizes
        if self.N <= 0 or self.T <= 0:
            raise ValueError(f"N and T must be positive, got N={self.N}, T={self.T}.")

        # Seeds
        if self.seeds_f0 < 0 or self.seeds_r0 < 0:
            raise ValueError("Seed counts must be >= 0.")
        if self.seeds_f0 + self.seeds_r0 > self.N * self.N:
            raise ValueError("Too many seeds for grid size.")

        # Scheme
        if self.update_scheme not in ("sync", "async"):
            raise ValueError("update_scheme must be 'sync' or 'async'.")

        # Probabilities in [0,1]
        prob_fields = [
            "beta_see", "beta_share_f", "beta_share_r",
            "gamma_correction", "gamma_switch",
            "delta_decay_f", "delta_decay_r",
            "eta_misclass", "spatial_strength",
        ]
        for name in prob_fields:
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}.")

        # Heterogeneity SD (>= 0)
        if float(self.hetero_sd) < 0.0:
            raise ValueError(f"hetero_sd must be >= 0, got {self.hetero_sd}.")

        # Refractory non-negative integer
        if not isinstance(self.tau_post, int) or self.tau_post < 0:
            raise ValueError(f"tau_post must be an integer >= 0, got {self.tau_post}.")

        # Spatial mode
        if self.spatial_mode not in {"radial", "x-gradient"}:
            raise ValueError(f"spatial_mode must be one of {{'radial','x-gradient'}}, got {self.spatial_mode!r}.")

        # RNG seed (optional)
        if self.rng_seed is not None:
            try:
                self.rng_seed = int(self.rng_seed)
            except Exception as e:
                raise ValueError(f"rng_seed must be castable to int, got {self.rng_seed!r}.") from e
            if not (0 <= self.rng_seed <= 2**32 - 1):
                raise ValueError(f"rng_seed must be in [0, 2**32-1], got {self.rng_seed}.")

        # Keep micro_async consistent with scheme
        self.micro_async = (self.update_scheme == "async")
