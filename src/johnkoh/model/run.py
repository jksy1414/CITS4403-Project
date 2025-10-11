import os, json, time
from .states import Params
from .simulate import simulate

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def main():
    # 1. set up parameters (you can tweak later)
    p = Params(
        N=50, T=60,
        seeds_f0=6, seeds_r0=5,       # tiny nudge to help early fake
        beta_see=0.30,                 # ↓ visibility to avoid 99% reach
        beta_share_f=0.58,             # fake catchier
        beta_share_r=0.46,             # real still spreads well
        gamma_correction=0.45,         # correction works but not crushing
        gamma_switch=0.12,             # fewer flips than 0.3
        delta_decay_f=0.12,            # fake fades faster
        delta_decay_r=0.06,
        rng_seed=None,  # <-- random each run
    )
    # 2. run the simulation
    out = simulate(p)

    # 3. attach metadata (useful for reproducibility)
    totals = {
        "shares_fake": int(sum(out.get("shares_f", []))),
        "shares_real": int(sum(out.get("shares_r", []))),
    }
    out = {
        **out,
        "meta": {
            "params": p.__dict__,
            "created_at": _timestamp()
        }, 
        "totals": totals,
    }

    # 4. save to JSON file
    os.makedirs("data/runs", exist_ok=True)
    filename = f"data/runs/CA_run_{_timestamp()}.json"
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {filename}")

    # 5. print friendly summary
    print("Results:")
    print(f"Seed used: {out.get('seed_used')}")
    print(f"Peak fake (posters): {max(out['I_f']) if out['I_f'] else 0}")
    print(f"Peak real (posters): {max(out['I_r']) if out['I_r'] else 0}")
    if "reach_fake" in out:
        print(f"Reach — Fake: {out['reach_fake']:.2%}, Real: {out['reach_real']:.2%}")
    print(f"Total shares — Fake: {totals['shares_fake']}, Real: {totals['shares_real']}")
    peak_f = max(out["I_f"]) if out["I_f"] else 0
    peak_r = max(out["I_r"]) if out["I_r"] else 0
    t_peak_f = out["I_f"].index(peak_f) if peak_f else None
    t_peak_r = out["I_r"].index(peak_r) if peak_r else None
    print(f"Time to peak — Fake: {t_peak_f}, Real: {t_peak_r}")

if __name__ == "__main__":
    main()
