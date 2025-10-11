from pathlib import Path
import os, json, time
from .states import Params
from .simulate import simulate

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _project_root() -> Path:
    # model/ -> johnkoh/ -> src/ -> project-root/
    return Path(__file__).resolve().parents[3]

def main():
    p = Params(
        N=50, T=60,
        seeds_f0=6, seeds_r0=5,
        beta_see=0.25,
        beta_share_f=0.60,
        beta_share_r=0.44,
        gamma_correction=0.35,
        gamma_switch=0.08,
        delta_decay_f=0.14,
        delta_decay_r=0.10,
        rng_seed=None,
    )
    out = simulate(p)

    totals = {
        "shares_fake": int(sum(out.get("shares_f", []))),
        "shares_real": int(sum(out.get("shares_r", []))),
    }
    out = {
        **out,
        "meta": {"params": p.__dict__, "created_at": _timestamp()},
        "totals": totals,
    }

    # NEW: save under project-root/data/johnkoh/runs
    root = _project_root()
    runs_dir = root / "data" / "johnkoh" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    filename = runs_dir / f"CA_run_{_timestamp()}.json"
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Results saved → {filename}")
    print("Results:")
    print(f"Seed used: {out.get('seed_used')}")
    peak_f = max(out["I_f"]) if out["I_f"] else 0
    peak_r = max(out["I_r"]) if out["I_r"] else 0
    print(f"Peak fake (posters): {peak_f}")
    print(f"Peak real (posters): {peak_r}")
    if "reach_fake" in out:
        print(f"Reach — Fake: {out['reach_fake']:.2%}, Real: {out['reach_real']:.2%}")
    print(f"Total shares — Fake: {totals['shares_fake']}, Real: {totals['shares_real']}")
    t_peak_f = out["I_f"].index(peak_f) if peak_f else None
    t_peak_r = out["I_r"].index(peak_r) if peak_r else None
    print(f"Time to peak — Fake: {t_peak_f}, Real: {t_peak_r}")

if __name__ == "__main__":
    main()
