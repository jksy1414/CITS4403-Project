from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# --- small helpers -----------------------------------------------------------

def _ensure_ax(ax, *, figsize=(7, 4)):
    """Return a Matplotlib Axes, creating one if None was passed."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _series(run: Dict[str, Any], key: str) -> Sequence[int]:
    """Pull a list-like series from a run dict, tolerating missing/None."""
    return run.get(key, []) or []


# ----- Macro: single-run time series ----------------------------------------

def plot_macro(run: Dict[str, Any], *, ax=None, title: str = "Macro Dynamics (I_f vs I_r)"):
    """
    Plot active posters over time for one run:
    I_f (fake) and I_r (real) on the same axes.
    """
    I_f: Sequence[int] = _series(run, "I_f")
    I_r: Sequence[int] = _series(run, "I_r")

    ax = _ensure_ax(ax, figsize=(7, 4))
    ax.plot(I_f, label="I_f (fake)")
    ax.plot(I_r, label="I_r (real)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Active posters")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax


def plot_shares(run: Dict[str, Any], *, ax=None, title: str = "New Shares per Step"):
    """
    Plot per-step new shares for fake and real streams.
    """
    S_f: Sequence[int] = _series(run, "shares_f")
    S_r: Sequence[int] = _series(run, "shares_r")

    ax = _ensure_ax(ax, figsize=(7, 4))
    ax.plot(S_f, label="shares_f")
    ax.plot(S_r, label="shares_r")
    ax.set_xlabel("Time step")
    ax.set_ylabel("New shares")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax


# ----- Macro: multi-run overlay ---------------------------------------------

def plot_multi_run_overlay(runs: Iterable[Dict[str, Any]], *, ax=None, title: str = "Randomness: Multi-run Overlay"):
    """
    Overlay many runs to show volatility/dispersion.
    Each run contributes its I_f and I_r traces (if present).
    """
    ax = _ensure_ax(ax, figsize=(7, 4))
    for r in runs:
        I_f = _series(r, "I_f")
        I_r = _series(r, "I_r")
        if I_f:
            ax.plot(I_f, alpha=0.35)
        if I_r:
            ax.plot(I_r, alpha=0.35)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Active posters")
    ax.set_title(title)
    ax.grid(True)
    return ax


# ----- Macro: reach summary --------------------------------------------------

def plot_reach_bar(
    runs: Iterable[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    *,
    ax=None,
    title: str = "Reach Comparison",
):
    """
    Bar chart comparing reach for each run (fake vs real).
    Input reach values may be proportions (0..1) or percentages; this
    function follows the original behavior and multiplies by 100 either way.
    """
    runs_list = list(runs)
    n = len(runs_list)
    if labels is None:
        labels = [f"run {i+1}" for i in range(n)]

    # Preserve original semantics: cast to float and multiply by 100
    reach_f = [float(r.get("reach_fake", 0.0) or 0.0) * 100 for r in runs_list]
    reach_r = [float(r.get("reach_real", 0.0) or 0.0) * 100 for r in runs_list]

    idx = np.arange(n)
    width = 0.35

    ax = _ensure_ax(ax, figsize=(max(6, 1.5 * n), 4))
    ax.bar(idx - width / 2, reach_f, width, label="reach_fake (%)")
    ax.bar(idx + width / 2, reach_r, width, label="reach_real (%)")
    ax.set_xticks(idx, labels)
    ax.set_ylabel("Reach (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return ax


# ----- Micro: grid snapshots -------------------------------------------------

def plot_snapshots(
    run: Dict[str, Any],
    order: Optional[List[str]] = None,
    *,
    axarr=None,
    cmap: str = "viridis",
    title: str = "Micro Snapshots",
):
    """
    Visualize stored grid snapshots at selected time keys.
    Expects run["grid_snapshots"] = { "t0": array, "t5": array, ... }.
    The color scale is fixed to [0, 4] to match the State enum.
    """
    snaps = run.get("grid_snapshots") or {}
    if not snaps:
        raise ValueError("No 'grid_snapshots' found. Enable snapshots in simulate() first.")

    # Respect caller order if provided; otherwise sort by numeric t
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
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=4)  # preserve exact vmin/vmax
        ax.set_title(k)
        ax.set_xticks([])
        ax.set_yticks([])

    if created_fig:
        # Preserve original colorbar behavior and layout
        plt.colorbar(im, ax=axarr if isinstance(axarr, np.ndarray) else [axarr], fraction=0.02)
        plt.suptitle(title)

    return axarr
