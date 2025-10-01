import numpy as np
from .states import State

# 8-neighbour (Moore) offsets: up, down, left, right + 4 diagonals
# Keeping it simple: we will ALWAYS use these in the minimal version.
_OFFSETS_MOORE = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def posting_neighbour_flags(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal version: toroidal wrap-around, 8-neighbour (Moore).
    Given a grid of ints (State codes), return two boolean grids:
      has_fake: True where at least one neighbour is posting FAKE (I_F)
      has_real: True where at least one neighbour is posting REAL (I_R)
    """
    # sanity
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square 2D array (N x N)")

    has_fake = np.zeros_like(state, dtype=bool)
    has_real = np.zeros_like(state, dtype=bool)

    # roll the grid in each neighbour direction (wrap-around)
    # and OR-accumulate where neighbours equal I_F or I_R
    for dx, dy in _OFFSETS_MOORE:
        rolled = np.roll(np.roll(state, dx, axis=0), dy, axis=1)
        has_fake |= (rolled == State.I_F)
        has_real |= (rolled == State.I_R)

    return has_fake, has_real
