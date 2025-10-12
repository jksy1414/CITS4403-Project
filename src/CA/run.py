# src/ca/run.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import re

from .states import Params
from .simulate import simulate
from utils.io import save_json
from utils.io_paths import runs_dir, timestamped_run_path

# External summariser is optional; we won't rely on it
try:
    from utils.metrics import summarise as _summarise_external
except Exception:
    _summarise_external = None


def _sanitize_label(text: str) -> str:
    t = text.strip().lower().replace(" ", "_")
    t = re.sub(r"[^a-z0-9_-]+", "", t)
    return t or "run"


def _make_batch_dir(model_name: str = "ca", label: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_dir(model_name)
    d = base / (f"{_sanitize_label(label)}_{ts}" if label else f"batch_{ts}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _derive_summary(run: dict) -> dict:
    """
    Built-in summary using proportions (0–1) for reach.
    This function never returns percentages for reach.
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    shares_f = run.get("shares_f", []) or []
    shares_r = run.get("shares_r", []) or []
    # The sim should already store proportions, but be defensive:
    reach_fake = float(run.get("reach_fake", 0.0) or 0.0)
    reach_real = float(run.get("reach_real", 0.0) or 0.0)
    if reach_fake > 1.0:  # normalize if percent slipped in
        reach_fake /= 100.0
    if reach_real > 1.0:
        reach_real /= 100.0

    peak_f = max(I_f) if I_f else 0
    peak_r = max(I_r) if I_r else 0
    t_peak_f = (I_f.index(peak_f) if I_f and (peak_f in I_f) else None)
    t_peak_r = (I_r.index(peak_r) if I_r and (peak_r in I_r) else None)

    return {
        "peak_f": peak_f,
        "peak_r": peak_r,
        "t_peak_f": t_peak_f,
        "t_peak_r": t_peak_r,
        "total_shares_f": int(sum(shares_f)),
        "total_shares_r": int(sum(shares_r)),
        "reach_fake": reach_fake,  # proportions (0–1)
        "reach_real": reach_real,  # proportions (0–1)
    }


def _normalize_reach_in_place(d: dict) -> None:
    """
    Ensure d['reach_fake'] and d['reach_real'] are proportions (0–1), even
    if an external summariser overwrote them with percentages (0–100).
    """
    for k in ("reach_fake", "reach_real"):
        if k in d and d[k] is not None:
            try:
                v = float(d[k])
                if v > 1.0:
                    d[k] = v / 100.0
                else:
                    d[k] = v
            except (TypeError, ValueError):
                d[k] = 0.0


def _print_single_summary(out: dict, path: Path) -> None:
    sm = _derive_summary(out)

    # If external summariser exists, merge then re-normalize reach
    if _summarise_external is not None:
        try:
            ext = _summarise_external(out) or {}
            sm.update(ext)
        finally:
            _normalize_reach_in_place(sm)

    print(f"Results saved → {path}")
    print("Results:")
    print(f"Seed used: {out.get('seed_used')}")
    print(f"Peak fake (posters): {sm['peak_f']}")
    print(f"Peak real (posters): {sm['peak_r']}")
    print(f"Reach — Fake: {sm['reach_fake']:.2%}, Real: {sm['reach_real']:.2%}")
    print(f"Total shares — Fake: {sm['total_shares_f']}, Real: {sm['total_shares_r']}")
    print(f"Time to peak — Fake: {sm['t_peak_f']}, Real: {sm['t_peak_r']}")


def main():
    parser = argparse.ArgumentParser(description="Run CA simulations (single or batch).")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulations to run (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed (optional). Batch uses seed+i.")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for batch folder name, e.g. 'baseline' or 'baseline_hetero'.")
    args = parser.parse_args()

    base_params = dict(
        N=50, T=60,
        seeds_f0=6, seeds_r0=5,
        beta_see=0.35, beta_share_f=0.50, beta_share_r=0.40,
        gamma_correction=0.7, gamma_switch=0.3,
        delta_decay_f=0.12, delta_decay_r=0.06,
    )

    # Single run
    if args.runs <= 1:
        p = Params(**base_params, rng_seed=args.seed)
        out = simulate(p)
        path = timestamped_run_path(model_name="ca", prefix="CA_run")
        save_json(out, path)
        _print_single_summary(out, path)
        return

    # Batch
    batch_dir = _make_batch_dir(model_name="ca", label=args.label)
    print(f"Batch dir → {batch_dir}")

    base_seed = args.seed
    if base_seed is None:
        import numpy as _np
        base_seed = int(_np.random.default_rng().integers(0, 2**32 - 1))

    rows = []

    for i in range(args.runs):
        run_seed = base_seed + i
        p = Params(**base_params, rng_seed=run_seed)
        out = simulate(p)

        run_path = batch_dir / f"CA_run_{i:02d}.json"
        save_json(out, run_path)

        # Build robust per-run summary
        sm = _derive_summary(out)

        # Merge external summariser (if available) then re-normalize reach
        if _summarise_external is not None:
            try:
                ext = _summarise_external(out) or {}
                sm.update(ext)
            finally:
                _normalize_reach_in_place(sm)

        # Bookkeeping
        sm["run_index"] = i
        sm["seed"] = out.get("seed_used", run_seed)
        sm["json_path"] = str(run_path)
        rows.append(sm)

        print(f"[{i+1}/{args.runs}] seed={sm['seed']} "
              f"peak_f={sm['peak_f']} peak_r={sm['peak_r']} "
              f"reachF={sm['reach_fake']:.2%} reachR={sm['reach_real']:.2%}")

    # Save CSV/JSON summaries
    try:
        import pandas as pd
        df = pd.DataFrame(rows)

        # Ensure the dataframe columns are proportions (0–1)
        for col in ("reach_fake", "reach_real"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                df.loc[df[col] > 1.0, col] = df.loc[df[col] > 1.0, col] / 100.0

        csv_path = batch_dir / "summary.csv"
        df.to_csv(csv_path, index=False)

        agg = {
            "runs": args.runs,
            "label": _sanitize_label(args.label) if args.label else None,
            "base_seed": base_seed,
            # Means/stds over proportions
            "peak_f_mean": float(df["peak_f"].mean()),
            "peak_r_mean": float(df["peak_r"].mean()),
            "reach_fake_mean": float(df["reach_fake"].mean()),
            "reach_real_mean": float(df["reach_real"].mean()),
            "total_shares_f_mean": float(df["total_shares_f"].mean()),
            "total_shares_r_mean": float(df["total_shares_r"].mean()),
            "peak_f_std": float(df["peak_f"].std(ddof=0)),
            "peak_r_std": float(df["peak_r"].std(ddof=0)),
            "reach_fake_std": float(df["reach_fake"].std(ddof=0)),
            "reach_real_std": float(df["reach_real"].std(ddof=0)),
        }
        save_json({"aggregate": agg, "rows": rows}, batch_dir / "summary.json")

        print("\nBatch summary:")
        print(f"CSV   → {csv_path}")
        print(f"JSON  → {batch_dir/'summary.json'}")
        print(f"Means → peak_f={agg['peak_f_mean']:.2f}, peak_r={agg['peak_r_mean']:.2f}, "
              f"reachF={agg['reach_fake_mean']:.2%}, reachR={agg['reach_real_mean']:.2%}")
    except Exception as e:
        print(f"Note: could not write summary.csv / summary.json ({e}). Per-run JSONs are saved in {batch_dir}.")


if __name__ == "__main__":
    main()
