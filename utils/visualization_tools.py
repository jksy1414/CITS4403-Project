from __future__ import annotations
from typing import Iterable, Dict, Any, List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

def show_time_series(run_data: Dict[str, Any], *, ax=None, title: str = "Active Sharers Over Time") -> plt.Axes:
    """
    Plot the number of active fake and real posters over time.
    """
    I_f = run_data.get("I_f", [])
    I_r = run_data.get("I_r", [])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(I_f, label="Fake (I_f)", color='tab:red')
    ax.plot(I_r, label="Real (I_r)", color='tab:green')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Active Users")
    ax.grid(True, linestyle=":")
    ax.legend()
    return ax

def show_new_shares(run_data: Dict[str, Any], *, ax=None, title: str = "Sharing Activity Per Step") -> plt.Axes:
    """
    Plot how many new fake and real posts were made at each time step.
    """
    shares_f = run_data.get("shares_f", [])
    shares_r = run_data.get("shares_r", [])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(shares_f, label="Fake shares", color='tab:orange')
    ax.plot(shares_r, label="Real shares", color='tab:blue')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("New Shares")
    ax.grid(True, linestyle=":")
    ax.legend()
    return ax

def compare_multiple_runs(runs: Iterable[Dict[str, Any]], *, ax=None, title: str = "Run Overlay (Randomness)") -> plt.Axes:
    """
    Overlay fake and real curves from multiple independent runs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for run in runs:
        I_f = run.get("I_f", [])
        I_r = run.get("I_r", [])
        if I_f:
            ax.plot(I_f, color='red', alpha=0.25)
        if I_r:
            ax.plot(I_r, color='green', alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Active Posters")
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax

def plot_reach_summary(runs: Iterable[Dict[str, Any]], labels: Optional[List[str]] = None, *, ax=None, title: str = "Total Reach per Run") -> plt.Axes:
    """
    Plot bar chart comparing fake vs real reach across multiple runs.
    """
    run_list = list(runs)
    count = len(run_list)
    if labels is None:
        labels = [f"Run {i+1}" for i in range(count)]

    reach_f = [float(r.get("reach_fake", 0.0) or 0.0) * 100 for r in run_list]
    reach_r = [float(r.get("reach_real", 0.0) or 0.0) * 100 for r in run_list]

    x = np.arange(count)
    width = 0.35
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, count * 1.6), 4))
    ax.bar(x - width / 2, reach_f, width, label="Fake (%)", color='salmon')
    ax.bar(x + width / 2, reach_r, width, label="Real (%)", color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reach (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    return ax

def show_grid_snapshots(run_data: Dict[str, Any], keys: Optional[List[str]] = None, *, cmap: str = "viridis", title: str = "Grid State Snapshots") -> List[plt.Axes]:
    """
    Display a sequence of grid snapshots stored in the simulation result.
    """
    snapshots = run_data.get("grid_snapshots", {})
    if not snapshots:
        raise ValueError("No snapshots found. Make sure they were recorded during simulation.")

    selected_keys = keys or sorted(snapshots.keys(), key=lambda k: int(k.lstrip("t")))
    num = len(selected_keys)

    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4), squeeze=False)
    axes = axes[0]

    for ax, k in zip(axes, selected_keys):
        grid = np.asarray(snapshots[k])
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(k)
        ax.axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    fig.suptitle(title)
    return axes.tolist()
