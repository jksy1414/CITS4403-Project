from __future__ import annotations
import numpy as np
from .states import State

def posting_neighbour_flags(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Moore-8 neighbour check with toroidal boundary.
    Returns two boolean arrays (saw_f, saw_r) of shape (N, N) indicating
    whether each cell sees at least one fake/real poster among its neighbours.
    """
    N = state.shape[0]
    I_F = (state == State.I_F)
    I_R = (state == State.I_R)

    # roll in 8 directions
    neigh_f = (
        np.roll(I_F, 1, 0) | np.roll(I_F, -1, 0) |
        np.roll(I_F, 1, 1) | np.roll(I_F, -1, 1) |
        np.roll(np.roll(I_F, 1, 0), 1, 1) |
        np.roll(np.roll(I_F, 1, 0), -1, 1) |
        np.roll(np.roll(I_F, -1, 0), 1, 1) |
        np.roll(np.roll(I_F, -1, 0), -1, 1)
    )
    neigh_r = (
        np.roll(I_R, 1, 0) | np.roll(I_R, -1, 0) |
        np.roll(I_R, 1, 1) | np.roll(I_R, -1, 1) |
        np.roll(np.roll(I_R, 1, 0), 1, 1) |
        np.roll(np.roll(I_R, 1, 0), -1, 1) |
        np.roll(np.roll(I_R, -1, 0), 1, 1) |
        np.roll(np.roll(I_R, -1, 0), -1, 1)
    )
    return neigh_f, neigh_r

# ---------- ASYNC helper below ----------

def _wrap(N: int, i: int, j: int) -> tuple[int, int]:
    return i % N, j % N

def posting_neighbour_flags_local(state: np.ndarray, i: int, j: int) -> tuple[bool, bool]:
    """
    For async mode: check the 8-neighbourhood around (i,j) against the *current*
    (possibly partially updated) state. Returns (saw_f, saw_r).
    """
    N = state.shape[0]
    saw_f = False
    saw_r = False
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = _wrap(N, i + di, j + dj)
            s = state[ii, jj]
            if s == State.I_F:
                saw_f = True
            elif s == State.I_R:
                saw_r = True
            if saw_f and saw_r:
                return True, True
    return saw_f, saw_r