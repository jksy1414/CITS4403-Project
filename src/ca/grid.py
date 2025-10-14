from __future__ import annotations
import numpy as np
from .states import State


# Define relative 8-neighbour offsets (Moore neighbourhood)
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


# ---------------------------------------------------------------------
# Vectorised neighbour detection (for synchronous updates)
# ---------------------------------------------------------------------

def posting_neighbour_flags(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Check whether each grid cell has at least one FAKE or REAL poster nearby.

    Uses an 8-neighbour (Moore) pattern with toroidal wrapping.
    Returns two boolean arrays (saw_f, saw_r) of shape (N, N).

    Complexity: O(N^2) per frame, small constant overhead.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError("state must be a NumPy ndarray.")
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square 2D array of shape (N, N).")

    I_F = (state == State.I_F)
    I_R = (state == State.I_R)

    # Shift arrays in all 8 directions (toroidal wrap)
    rolled_fakes = [np.roll(np.roll(I_F, dx, axis=0), dy, axis=1) for dx, dy in _OFFSETS]
    rolled_reals = [np.roll(np.roll(I_R, dx, axis=0), dy, axis=1) for dx, dy in _OFFSETS]

    saw_f = np.logical_or.reduce(rolled_fakes) if rolled_fakes else np.zeros_like(I_F, dtype=bool)
    saw_r = np.logical_or.reduce(rolled_reals) if rolled_reals else np.zeros_like(I_R, dtype=bool)

    return saw_f, saw_r


# ---------------------------------------------------------------------
# Local neighbour detection (for asynchronous updates)
# ---------------------------------------------------------------------

def _wrap(N: int, i: int, j: int) -> tuple[int, int]:
    """Wrap coordinates (i, j) into [0, N) range using toroidal topology."""
    return i % N, j % N


def posting_neighbour_flags_local(state: np.ndarray, i: int, j: int) -> tuple[bool, bool]:
    """
    Check if the cell at (i, j) sees any FAKE or REAL poster among its 8 neighbours.
    Used during asynchronous (in-place) updates.

    Returns:
        (saw_f, saw_r) : booleans for each neighbour type.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError("state must be a NumPy ndarray.")
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square 2D array of shape (N, N).")

    N = state.shape[0]
    saw_f = False
    saw_r = False

    for dx, dy in _OFFSETS:
        ni, nj = _wrap(N, i + dx, j + dy)
        s = state[ni, nj]
        if s == State.I_F:
            saw_f = True
        elif s == State.I_R:
            saw_r = True
        if saw_f and saw_r:
            break

    return saw_f, saw_r
