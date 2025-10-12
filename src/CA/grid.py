# src/ca/grid.py
import numpy as np
from .states import State

# returns two boolean arrays: saw_f, saw_r (True if any neighbour is posting F/R)
def posting_neighbour_flags(state: np.ndarray):
    N = state.shape[0]
    is_IF = (state == State.I_F)
    is_IR = (state == State.I_R)

    def any_neigh(mask):
        acc = np.zeros_like(mask, dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                acc |= np.roll(np.roll(mask, di, axis=0), dj, axis=1)
        return acc

    saw_f = any_neigh(is_IF)
    saw_r = any_neigh(is_IR)
    return saw_f, saw_r
