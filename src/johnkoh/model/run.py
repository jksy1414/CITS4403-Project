import os, json, time
from .states import Params
from .simulate import simulate

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def main():
    # 1. set up parameters (you can tweak later)
    p = Params(
        N=50, T=20,
        seeds_f0=5, seeds_r0=5,
        beta_see=0.35, beta_share_f=0.50, beta_share_r=0.40,
        gamma_correction=0.7, gamma_switch=0.3,
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

if __name__ == "__main__":
    main()
