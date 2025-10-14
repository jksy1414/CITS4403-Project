from __future__ import annotations
import numpy as np
from .states import State

# Moore neighbourhood (8 directions) as (di, dj) offsets
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


def _validate_square_array(a: np.ndarray) -> None:
    """Raise early if `a` is not a square 2-D numpy array."""
    if not isinstance(a, np.ndarray):
        raise TypeError("state must be a numpy.ndarray")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("state must be a square 2-D array of shape (N, N)")


def posting_neighbour_flags(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Moore-8 neighbour check with toroidal boundary (wrap-around).
    Returns two boolean arrays (saw_f, saw_r) of shape (N, N) indicating
    whether each cell sees at least one FAKE/REAL poster among its neighbours,
    based on the snapshot 'state'.

    Complexity: O(N^2) per call with small constant factors.
    """
    _validate_square_array(state)

    # Posters at this snapshot
    I_F = (state == State.I_F)
    I_R = (state == State.I_R)

    # Build the eight wrapped views and OR-reduce them into visibility flags.
    rolls_f = [np.roll(np.roll(I_F, di, axis=0), dj, axis=1) for di, dj in _OFFSETS]
    rolls_r = [np.roll(np.roll(I_R, di, axis=0), dj, axis=1) for di, dj in _OFFSETS]

    # logical_or.reduce guarantees boolean dtype in output
    saw_f = np.logical_or.reduce(rolls_f) if rolls_f else np.zeros_like(I_F, dtype=bool)
    saw_r = np.logical_or.reduce(rolls_r) if rolls_r else np.zeros_like(I_R, dtype=bool)

    return saw_f, saw_r


# ---------- ASYNC helpers below ----------

def _wrap(N: int, i: int, j: int) -> tuple[int, int]:
    """Toroidal wrapping of indices (i, j) into range [0, N)."""
    return i % N, j % N


def posting_neighbour_flags_local(state: np.ndarray, i: int, j: int) -> tuple[bool, bool]:
    """
    For async mode: check the 8-neighbourhood around (i, j) against the *current*
    (possibly partially updated) state. Returns (saw_f, saw_r).

    This is O(1) per query and should be used during per-cell updates when
    the grid is being updated in-place.
    """
    _validate_square_array(state)

    N = state.shape[0]
    saw_f = False
    saw_r = False

    for di, dj in _OFFSETS:
        ii, jj = _wrap(N, i + di, j + dj)
        s = state[ii, jj]
        if s == State.I_F:
            saw_f = True
        elif s == State.I_R:
            saw_r = True
        if saw_f and saw_r:
            return True, True

    return saw_f, saw_r
