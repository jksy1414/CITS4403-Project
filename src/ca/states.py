from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class State(IntEnum):
    """
    Core agent states for the Cellular Automata grid.

    0 - Susceptible (S): has not seen any information
    1 - Exposed to Fake (E_F): seen fake, not yet sharing
    2 - Exposed to Real (E_R): seen real, not yet sharing
    3 - Posting Fake (I_F): currently spreading fake
    4 - Posting Real (I_R): currently spreading real
    """
    S = 0
    E_F = 1
    E_R = 2
    I_F = 3
    I_R = 4


@dataclass
class Params:
    """
    Parameter configuration for the CA simulation.

    All values come with sensible defaults but can be overridden
    at runtime to experiment with different behaviours.
    """

    # --- Grid and timing ---
    N: int = 50                  # Grid size (NxN)
    T: int = 60                  # Number of time steps

    # --- Initial seeding ---
    seeds_f0: int = 20
    seeds_r0: int = 10

    # --- Behaviour probabilities ---
    beta_see: float = 0.35
    beta_share_f: float = 0.50
    beta_share_r: float = 0.40

    # --- Misclassification / correction ---
    gamma_correction: float = 0.70
    gamma_switch: float = 0.30

    # --- Decay of engagement ---
    delta_decay_f: float = 0.12
    delta_decay_r: float = 0.06

    # --- Randomness and update control ---
    rng_seed: Optional[int] = None
    update_scheme: str = "sync"  # "sync" or "async"

    # --- Micro-level mechanisms ---
    micro_async: bool = False
    micro_refractory: bool = False
    micro_misclass: bool = False

    # --- Macro-level mechanisms ---
    macro_hetero: bool = False
    macro_spatial: bool = False

    # --- Refractory behaviour ---
    tau_post: int = 3

    # --- Misclassification tuning ---
    eta_misclass: float = 0.02

    # --- Macro configuration ---
    hetero_sd: float = 0.20
    spatial_strength: float = 0.35
    spatial_mode: str = "radial"

    def __post_init__(self):
        """Validate parameter ranges and ensure safe configuration."""
        # Grid size and duration
        if self.N <= 0 or self.T <= 0:
            raise ValueError(f"N and T must be positive. Got N={self.N}, T={self.T}.")

        # Seed validation
        if self.seeds_f0 < 0 or self.seeds_r0 < 0:
            raise ValueError("Seed counts must be >= 0.")
        if self.seeds_f0 + self.seeds_r0 > self.N * self.N:
            raise ValueError("Too many seeds for grid size.")

        # Update scheme
        if self.update_scheme not in ("sync", "async"):
            raise ValueError("update_scheme must be 'sync' or 'async'.")

        # Validate probability-type fields
        prob_fields = [
            "beta_see", "beta_share_f", "beta_share_r",
            "gamma_correction", "gamma_switch",
            "delta_decay_f", "delta_decay_r",
            "eta_misclass", "spatial_strength"
        ]
        for name in prob_fields:
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be within [0,1]. Got {v}.")

        # Validate heterogeneity SD
        if float(self.hetero_sd) < 0.0:
            raise ValueError(f"hetero_sd must be >= 0, got {self.hetero_sd}.")

        # Refractory parameter must be non-negative integer
        if not isinstance(self.tau_post, int) or self.tau_post < 0:
            raise ValueError(f"tau_post must be an integer >= 0, got {self.tau_post}.")

        # Spatial mode check
        if self.spatial_mode not in {"radial", "x-gradient"}:
            raise ValueError(
                f"spatial_mode must be 'radial' or 'x-gradient', got {self.spatial_mode!r}."
            )

        # RNG validation
        if self.rng_seed is not None:
            try:
                self.rng_seed = int(self.rng_seed)
            except Exception as e:
                raise ValueError(f"rng_seed must be castable to int, got {self.rng_seed!r}.") from e
            if not (0 <= self.rng_seed <= 2**32 - 1):
                raise ValueError(f"rng_seed must be within [0, 2**32-1], got {self.rng_seed}.")

        # Keep async flag consistent with update scheme
        self.micro_async = (self.update_scheme == "async")
