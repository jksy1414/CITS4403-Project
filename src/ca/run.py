from __future__ import annotations
import argparse
import re
from datetime import datetime
from pathlib import Path

from .states import Params
from .simulate import simulate

# Updated util imports (reflecting your new filenames)
from utils.json_utils import save_json
from utils.path_utils import runs_dir, timestamped_run_path

try:
    from utils.metrics_tools import summarise as summarise_external
except Exception:
    summarise_external = None


# ---------------------------------------------------------------------
# Label and feature utilities
# ---------------------------------------------------------------------

def clean_label(text: str) -> str:
    """Standardize a string for filenames or directories."""
    label = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]+", "", label) or "run"


def describe_features(params: Params | dict) -> str:
    """Generate a readable tag summarizing which features are enabled."""
    src = vars(params) if isinstance(params, Params) else params
    tags = []
    if src.get("update_scheme") == "async":
        tags.append("async")
    if src.get("micro_refractory"):
        tags.append("refractory")
    if src.get("micro_misclass"):
        tags.append("misclass")
    if src.get("macro_hetero"):
        tags.append("hetero")
    if src.get("macro_spatial"):
        tags.append("spatial")
    return "_".join(tags) if tags else "baseline"


def auto_label(base: str | None, params: Params) -> str:
    """Append the feature suffix to a base label."""
    suffix = describe_features(params)
    return f"{clean_label(base)}_{suffix}" if base else suffix


def create_batch_dir(model_name: str = "ca", label: str | None = None) -> Path:
    """Create a timestamped directory to store batch runs."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_dir(model_name)
    name = f"{clean_label(label)}_{timestamp}" if label else f"batch_{timestamp}"
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# Result summarisation
# ---------------------------------------------------------------------

def summarize_run(result: dict) -> dict:
    """Compute core statistics (peaks, reach, totals) from simulation output."""
    I_f, I_r = result.get("I_f", []), result.get("I_r", [])
    shares_f, shares_r = result.get("shares_f", []), result.get("shares_r", [])
    reach_f, reach_r = float(result.get("reach_fake", 0.0)), float(result.get("reach_real", 0.0))

    # Normalize reach (percentage to proportion)
    if reach_f > 1.0: reach_f /= 100.0
    if reach_r > 1.0: reach_r /= 100.0

    peak_f = max(I_f) if I_f else 0
    peak_r = max(I_r) if I_r else 0
    t_peak_f = I_f.index(peak_f) if peak_f in I_f else None
    t_peak_r = I_r.index(peak_r) if peak_r in I_r else None

    return {
        "peak_f": peak_f, "peak_r": peak_r,
        "t_peak_f": t_peak_f, "t_peak_r": t_peak_r,
        "total_shares_f": int(sum(shares_f)),
        "total_shares_r": int(sum(shares_r)),
        "reach_fake": reach_f, "reach_real": reach_r,
    }


def print_summary(result: dict, path: Path) -> None:
    """Display concise metrics after each run."""
    summary = summarize_run(result)
    if summarise_external:
        try:
            summary.update(summarise_external(result) or {})
        finally:
            for k in ("reach_fake", "reach_real"):
                if summary.get(k, 0) > 1.0:
                    summary[k] /= 100.0

    print(f"Results saved → {path}")
    print("Summary:")
    print(f"Features: {describe_features(result.get('params', {}))}")
    print(f"Seed used: {result.get('seed_used')}")
    print(f"Peak fake: {summary['peak_f']}, Peak real: {summary['peak_r']}")
    print(f"Reach — Fake: {summary['reach_fake']:.2%}, Real: {summary['reach_real']:.2%}")
    print(f"Total shares — Fake: {summary['total_shares_f']}, Real: {summary['total_shares_r']}")
    print(f"Time to peak — Fake: {summary['t_peak_f']}, Real: {summary['t_peak_r']}")


# ---------------------------------------------------------------------
# Parameter flattening and metadata
# ---------------------------------------------------------------------

def flatten_params(p: Params) -> dict:
    """Convert Params dataclass to flat dictionary with 'param_' prefixes."""
    return {f"param_{k}": v for k, v in vars(p).items()}


def metadata_row(p: Params, label: str, features: str, seed: int, idx: int, path: Path) -> dict:
    meta = {
        "label": label, "features": features, "update_scheme": p.update_scheme,
        "run_index": idx, "seed": seed, "json_path": str(path)
    }
    meta.update(flatten_params(p))
    return meta


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CA simulation (single or batch mode).")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--scheme", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--micro", type=str, default="", help="Comma-separated micro toggles: async,refractory,misclass")
    parser.add_argument("--macro", type=str, default="", help="Comma-separated macro toggles: hetero,spatial")
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--hetero-sd", type=float, default=0.20)
    parser.add_argument("--spatial-strength", type=float, default=0.35)
    args = parser.parse_args()

    micro = {x.strip().lower() for x in args.micro.split(",") if x.strip()}
    macro = {x.strip().lower() for x in args.macro.split(",") if x.strip()}

    base_params = dict(
        N=50, T=60, seeds_f0=6, seeds_r0=5,
        beta_see=0.35, beta_share_f=0.50, beta_share_r=0.40,
        gamma_correction=0.7, gamma_switch=0.3,
        delta_decay_f=0.12, delta_decay_r=0.06
    )

    def build_params(seed: int | None) -> Params:
        return Params(
            **base_params,
            rng_seed=seed,
            update_scheme=args.scheme,
            micro_async=("async" in micro) or (args.scheme == "async"),
            micro_refractory=("refractory" in micro),
            micro_misclass=("misclass" in micro),
            macro_hetero=("hetero" in macro),
            macro_spatial=("spatial" in macro),
            tau_post=3, eta_misclass=args.eta,
            hetero_sd=args.hetero_sd,
            spatial_strength=args.spatial_strength,
            spatial_mode="radial",
        )

    proto = build_params(args.seed)
    features = describe_features(proto)
    label = auto_label(args.label, proto)

    # --- Single run ---
    if args.runs <= 1:
        p = build_params(args.seed)
        result = simulate(p)
        feature_tag = describe_features(result.get("params", {}))
        path = timestamped_run_path(model="ca", prefix=f"CA_{feature_tag}")
        save_json(result, path)
        print_summary(result, path)

        summary = summarize_run(result)
        if summarise_external:
            try:
                summary.update(summarise_external(result) or {})
            finally:
                for k in ("reach_fake", "reach_real"):
                    if summary.get(k, 0) > 1.0:
                        summary[k] /= 100.0

        summary.update(metadata_row(p, label, features, result.get("seed_used", args.seed), 0, path))
        save_json({"aggregate": None, "rows": [summary]}, path.with_name(path.stem + "_summary.json"))
        return

    # --- Batch run ---
    import numpy as np
    batch_dir = create_batch_dir("ca", label)
    print(f"Batch output → {batch_dir}")

    base_seed = args.seed or int(np.random.default_rng().integers(0, 2**32 - 1))
    rows = []

    for i in range(args.runs):
        seed = base_seed + i
        p = build_params(seed)
        result = simulate(p)
        run_path = batch_dir / f"CA_run_{i:02d}.json"
        save_json(result, run_path)

        summary = summarize_run(result)
        if summarise_external:
            try:
                summary.update(summarise_external(result) or {})
            finally:
                for k in ("reach_fake", "reach_real"):
                    if summary.get(k, 0) > 1.0:
                        summary[k] /= 100.0

        summary.update(metadata_row(p, label, features, result.get("seed_used", seed), i, run_path))
        rows.append(summary)

        print(
            f"[{i+1}/{args.runs}] seed={seed} "
            f"peakF={summary['peak_f']} peakR={summary['peak_r']} "
            f"reachF={summary['reach_fake']:.2%} reachR={summary['reach_real']:.2%} [{features}]"
        )

    # Save batch summaries
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        for col in ("reach_fake", "reach_real"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df.loc[df[col] > 1.0, col] /= 100.0

        df.to_csv(batch_dir / "summary.csv", index=False)
        agg = {
            "runs": args.runs, "label": label, "features": features,
            "base_seed": base_seed,
            "peak_f_mean": df["peak_f"].mean(), "peak_r_mean": df["peak_r"].mean(),
            "reach_fake_mean": df["reach_fake"].mean(), "reach_real_mean": df["reach_real"].mean(),
            "total_shares_f_mean": df["total_shares_f"].mean(),
            "total_shares_r_mean": df["total_shares_r"].mean(),
            "params_template": flatten_params(proto),
        }
        save_json({"aggregate": agg, "rows": rows}, batch_dir / "summary.json")

        print("\n[✓] Batch Summary:")
        print(f"CSV  → {batch_dir/'summary.csv'}")
        print(f"JSON → {batch_dir/'summary.json'}")
    except Exception as e:
        print(f"[!] Could not save summary: {e}")


if __name__ == "__main__":
    main()
