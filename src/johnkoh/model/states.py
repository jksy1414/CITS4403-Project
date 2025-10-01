from enum import IntEnum
from dataclasses import dataclass

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
    N: int = 100      # grid N x N (population size for CA)
    T: int = 200      # number of time steps
    rng_seed: int = 42

    # Behaviour (simple first)
    beta_see: float = 0.35
    beta_share_f: float = 0.50
    beta_share_r: float = 0.40

    # Initial posters
    seeds_f0: int = 20
    seeds_r0: int = 10
