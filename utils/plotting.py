from __future__ import annotations
from typing import Iterable, Dict, Any, List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

# ----- Macro: single-run time series -----

def plot_macro(run: Dict[str, Any], *, ax=None, title: str = "Macro Dynamics (I_f vs I_r)"):
    I_f: Sequence[int] = run.get("I_f", []) or []
    I_r: Sequence[int] = run.get("I_r", []) or []
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(I_f, label="I_f (fake)")
    ax.plot(I_r, label="I_r (real)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Active posters")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax

def plot_shares(run: Dict[str, Any], *, ax=None, title: str = "New Shares per Step"):
    S_f: Sequence[int] = run.get("shares_f", []) or []
    S_r: Sequence[int] = run.get("shares_r", []) or []
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(S_f, label="shares_f")
    ax.plot(S_r, label="shares_r")
    ax.set_xlabel("Time step")
    ax.set_ylabel("New shares")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax

# ----- Macro: multi-run overlay -----

def plot_multi_run_overlay(runs: Iterable[Dict[str, Any]], *, ax=None, title: str = "Randomness: Multi-run Overlay"):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        I_f = r.get("I_f", []) or []
        I_r = r.get("I_r", []) or []
        if I_f:
            ax.plot(I_f, alpha=0.35)
        if I_r:
            ax.plot(I_r, alpha=0.35)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Active posters")
    ax.set_title(title)
    ax.grid(True)
    return ax

# ----- Macro: reach summary -----

def plot_reach_bar(runs: Iterable[Dict[str, Any]], labels: Optional[List[str]] = None, *, ax=None, title: str = "Reach Comparison"):
    runs_list = list(runs)
    n = len(runs_list)
    if labels is None:
        labels = [f"run {i+1}" for i in range(n)]

    reach_f = [float(r.get("reach_fake", 0.0) or 0.0) * 100 for r in runs_list]
    reach_r = [float(r.get("reach_real", 0.0) or 0.0) * 100 for r in runs_list]

    idx = np.arange(n)
    width = 0.35
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.5 * n), 4))
    ax.bar(idx - width / 2, reach_f, width, label="reach_fake (%)")
    ax.bar(idx + width / 2, reach_r, width, label="reach_real (%)")
    ax.set_xticks(idx, labels)
    ax.set_ylabel("Reach (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return ax

# ----- Micro: grid snapshots -----

def plot_snapshots(run: Dict[str, Any], order: Optional[List[str]] = None, *, axarr=None, cmap: str = "viridis", title: str = "Micro Snapshots"):
    snaps = run.get("grid_snapshots") or {}
    if not snaps:
        raise ValueError("No 'grid_snapshots' found. Enable snapshots in simulate() first.")

    keys = order if order else sorted(snaps.keys(), key=lambda k: int(k.lstrip("t")))
    n = len(keys)

    created_fig = False
    if axarr is None:
        cols, rows = n, 1
        _, axarr = plt.subplots(rows, cols, figsize=(4 * cols, 4))
        if n == 1:
            axarr = [axarr]
        created_fig = True

    im = None
    for ax, k in zip(axarr, keys):
        arr = np.asarray(snaps[k])
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=4)  # match State enum 0..4
        ax.set_title(k)
        ax.set_xticks([]); ax.set_yticks([])

    if created_fig:
        plt.colorbar(im, ax=axarr if isinstance(axarr, np.ndarray) else [axarr], fraction=0.02)
        plt.suptitle(title)
    return axarr
