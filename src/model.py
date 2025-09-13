from dataclasses import dataclass
from typing import Literal

# User states per contagion (ints are fast and easy to count)
STATE_UNAWARE, STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE = range(5)

# Contagion labels (two competing items)
CONTAGION_FAKE, CONTAGION_REAL = "F", "R"

@dataclass
class Params:
    # Network / topology
    topology: Literal["ER","WS","BA"] = "BA"
    n: int = 2000
    mean_degree: int = 8
    p_er: float = 0.0         # if 0 → auto: mean_degree/(n-1)
    ws_rewire_p: float = 0.05
    ba_m: int = 4             # BA average degree ≈ 2m

    # Diffusion
    beta_see: float = 0.6     # see a neighbor's post
    beta_share: float = 0.20  # share after seeing
    decay: float = 0.05       # lose interest

    # Platform / competition
    flag_prob: float = 0.10         # fake only
    salience_fake: float = 1.20     # fake gets attention edge under limit
    correction_effect: float = 0.5  # seeing real reduces fake-sharing prob

    # Heterogeneity (per-user credulity)
    enable_heterogeneity: bool = False
    cred_alpha: float = 2.0
    cred_beta: float = 5.0

    # Simulation frame
    T: int = 60
    n_seeds_fake: int = 5
    n_seeds_real: int = 5
    limit_attention: bool = True
