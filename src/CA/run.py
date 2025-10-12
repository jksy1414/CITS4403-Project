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


# --------------------
# Label helpers
# --------------------
def _sanitize_label(text: str) -> str:
    t = text.strip().lower().replace(" ", "_")
    t = re.sub(r"[^a-z0-9_-]+", "", t)
    return t or "run"

def _features_suffix(p: "Params") -> str:
    """Create a label suffix from toggles/scheme."""
    tags = []
    if p.update_scheme == "async":
        tags.append("async")
    if getattr(p, "micro_refractory", False): tags.append("refractory")
    if getattr(p, "micro_misclass", False):   tags.append("misclass")
    if getattr(p, "micro_broadcast", False):  tags.append("broadcast")
    if getattr(p, "macro_hetero", False):     tags.append("hetero")
    if getattr(p, "macro_spatial", False):    tags.append("spatial")
    return "_".join(tags) if tags else "baseline"

def _auto_label(base_label: str | None, p: "Params") -> str:
    """If user didn’t pass --label, compose one from toggles/scheme."""
    suffix = _features_suffix(p)
    if base_label:
        return f"{_sanitize_label(base_label)}_{suffix}"
    return suffix

def _make_batch_dir(model_name: str = "ca", label: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_dir(model_name)
    d = base / (f"{_sanitize_label(label)}_{ts}" if label else f"batch_{ts}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------
# Summary helpers
# --------------------
def _derive_summary(run: dict) -> dict:
    """
    Built-in summary using proportions (0–1) for reach.
    This function never returns percentages for reach.
    """
    I_f = run.get("I_f", []) or []
    I_r = run.get("I_r", []) or []
    shares_f = run.get("shares_f", []) or []
    shares_r = run.get("shares_r", []) or []
    reach_fake = float(run.get("reach_fake", 0.0) or 0.0)
    reach_real = float(run.get("reach_real", 0.0) or 0.0)
    if reach_fake > 1.0: reach_fake /= 100.0
    if reach_real > 1.0: reach_real /= 100.0

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
    """Ensure reach_* are proportions (0–1) even if external summariser returns %."""
    for k in ("reach_fake", "reach_real"):
        if k in d and d[k] is not None:
            try:
                v = float(d[k])
                d[k] = v/100.0 if v > 1.0 else v
            except (TypeError, ValueError):
                d[k] = 0.0

def _print_single_summary(out: dict, path: Path) -> None:
    sm = _derive_summary(out)
    # Merge external summariser and re-normalize
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


def _params_to_columns(p: "Params") -> dict:
    """Flatten Params into CSV/row-friendly columns with 'param_' prefix."""
    d = dict(p.__dict__)
    # Don't duplicate rng_seed per row (we add a per-run 'seed' separately)
    # but keep it here for completeness; users sometimes want to see both.
    out = {}
    for k, v in d.items():
        out[f"param_{k}"] = v
    return out

def _row_metadata(p: "Params", label: str, features: str, seed_used: int, run_index: int, json_path: Path) -> dict:
    meta = {
        "label": label,
        "features": features,
        "update_scheme": p.update_scheme,
        "run_index": run_index,
        "seed": seed_used,
        "json_path": str(json_path),
    }
    meta.update(_params_to_columns(p))
    return meta


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Run CA simulations (single or batch).")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulations to run (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed (optional). Batch uses seed+i.")
    parser.add_argument("--label", type=str, default=None, help="Custom prefix for batch folder (optional).")
    parser.add_argument("--scheme", type=str, default="sync", choices=["sync","async"],
                        help="Update scheme (sync or async).")
    parser.add_argument("--micro", type=str, default="",
                        help="Comma-separated micro toggles: async,refractory,misclass,broadcast")
    parser.add_argument("--macro", type=str, default="",
                        help="Comma-separated macro toggles: hetero,spatial")
    args = parser.parse_args()

    # parse toggles
    micro_flags = {x.strip().lower() for x in args.micro.split(",") if x.strip()}
    macro_flags = {x.strip().lower() for x in args.macro.split(",") if x.strip()}

    # base parameters (unchanged)
    base_params = dict(
        N=50, T=60,
        seeds_f0=6, seeds_r0=5,
        beta_see=0.35, beta_share_f=0.50, beta_share_r=0.40,
        gamma_correction=0.7, gamma_switch=0.3,
        delta_decay_f=0.12, delta_decay_r=0.06,
    )

    # builder to create Params for a given seed
    def _build_params(seed: int | None) -> "Params":
        p = Params(
            **base_params,
            rng_seed=seed,
            update_scheme=args.scheme,
            # toggles (no behavior attached yet except async via update_scheme)
            micro_async=("async" in micro_flags) or (args.scheme == "async"),
            micro_refractory=("refractory" in micro_flags),
            micro_misclass=("misclass" in micro_flags),
            micro_broadcast=("broadcast" in micro_flags),
            macro_hetero=("hetero" in macro_flags),
            macro_spatial=("spatial" in macro_flags),
        )
        return p

    # proto params for naming (consistent folder label)
    proto_p = _build_params(args.seed)
    features = _features_suffix(proto_p)
    effective_label = _auto_label(args.label, proto_p)

    # single run
    if args.runs <= 1:
        p = _build_params(args.seed)
        out = simulate(p)
        path = timestamped_run_path(model_name="ca", prefix="CA_run")
        save_json(out, path)
        _print_single_summary(out, path)

        # Also write a compact single-run summary with config for auditing
        sm = _derive_summary(out)
        if _summarise_external is not None:
            try:
                ext = _summarise_external(out) or {}
                sm.update(ext)
            finally:
                _normalize_reach_in_place(sm)

        sm.update(_row_metadata(p, effective_label, features, out.get("seed_used", args.seed), 0, path))
        # Save next to the run file
        save_json({"aggregate": None, "rows": [sm]}, Path(str(path).replace(".json", "_summary.json")))
        return

    # batch
    batch_dir = _make_batch_dir(model_name="ca", label=effective_label)
    print(f"Batch dir → {batch_dir}")

    base_seed = args.seed
    if base_seed is None:
        import numpy as _np
        base_seed = int(_np.random.default_rng().integers(0, 2**32 - 1))

    rows = []

    for i in range(args.runs):
        run_seed = base_seed + i
        p = _build_params(run_seed)
        out = simulate(p)

        run_path = batch_dir / f"CA_run_{i:02d}.json"
        save_json(out, run_path)

        # robust per-run summary
        sm = _derive_summary(out)
        if _summarise_external is not None:
            try:
                ext = _summarise_external(out) or {}
                sm.update(ext)
            finally:
                _normalize_reach_in_place(sm)

        seed_used = out.get("seed_used", run_seed)
        sm.update(_row_metadata(p, effective_label, features, seed_used, i, run_path))
        rows.append(sm)

        print(f"[{i+1}/{args.runs}] seed={seed_used} "
              f"peak_f={sm['peak_f']} peak_r={sm['peak_r']} "
              f"reachF={sm['reach_fake']:.2%} reachR={sm['reach_real']:.2%} "
              f"[{features}]")

    # Save CSV/JSON summaries
    try:
        import pandas as pd
        df = pd.DataFrame(rows)

        # Ensure reach_* are proportions
        for col in ("reach_fake", "reach_real"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                df.loc[df[col] > 1.0, col] = df.loc[df[col] > 1.0, col] / 100.0

        csv_path = batch_dir / "summary.csv"
        df.to_csv(csv_path, index=False)

        agg = {
            "runs": args.runs,
            "label": effective_label,
            "features": features,
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
            # NEW: timing aggregates
            "t_peak_f_mean": float(pd.to_numeric(df["t_peak_f"], errors="coerce").mean()),
            "t_peak_r_mean": float(pd.to_numeric(df["t_peak_r"], errors="coerce").mean()),
            "t_peak_f_std": float(pd.to_numeric(df["t_peak_f"], errors="coerce").std(ddof=0)),
            "t_peak_r_std": float(pd.to_numeric(df["t_peak_r"], errors="coerce").std(ddof=0)),
            # Helpful snapshot of the config template that produced rows
            "params_template": _params_to_columns(proto_p),
        }
        save_json({"aggregate": agg, "rows": rows}, batch_dir / "summary.json")

        print("\nBatch summary:")
        print(f"CSV   → {csv_path}")
        print(f"JSON  → {batch_dir/'summary.json'}")
        print(f"Means → peak_f={agg['peak_f_mean']:.2f}, peak_r={agg['peak_r_mean']:.2f}, "
              f"reachF={agg['reach_fake_mean']:.2%}, reachR={agg['reach_real_mean']:.2%}")
        print(f"Times → t_peak_f≈{agg['t_peak_f_mean']:.1f}±{agg['t_peak_f_std']:.1f}, "
              f"t_peak_r≈{agg['t_peak_r_mean']:.1f}±{agg['t_peak_r_std']:.1f}  [{features}]")
    except Exception as e:
        print(f"Note: could not write summary.csv / summary.json ({e}). Per-run JSONs are saved in {batch_dir}.")


if __name__ == "__main__":
    main()
