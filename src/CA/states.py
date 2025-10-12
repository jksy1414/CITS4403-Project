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

# Minimal parameters for a first working baseline
@dataclass
class Params:
    # World / timing
    N: int = 50      # grid N x N (population size for CA)
    T: int = 60      # number of time steps
    rng_seed: Optional[int] = None  # None => random seed each run

    # Behaviour probabilities
    beta_see: float = 0.35
    beta_share_f: float = 0.50
    beta_share_r: float = 0.40

    # Initial posters conditions
    seeds_f0: int = 20
    seeds_r0: int = 10

     # Correction / switch mechanics
    gamma_correction: float = 0.7   # seeing real lowers chance to share fake
    gamma_switch: float = 0.3       # posters of fake may switch to real

    # Natural decay of attention (forgetting)
    delta_decay_f: float = 0.12  # fake forgets faster
    delta_decay_r: float = 0.06  # real forgets slower

    def __post_init__(self):
        if self.N <= 0 or self.T <= 0:
            raise ValueError("N and T must be positive.")
        if self.seeds_f0 < 0 or self.seeds_r0 < 0:
            raise ValueError("seed counts must be >= 0.")
        if self.seeds_f0 + self.seeds_r0 > self.N * self.N:
            raise ValueError("too many seeds for grid size.")
        for name in [
            "beta_see","beta_share_f","beta_share_r",
            "gamma_correction","gamma_switch","delta_decay_f","delta_decay_r"
        ]:
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}.")