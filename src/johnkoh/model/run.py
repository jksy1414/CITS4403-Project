import os, json, time
from .states import Params
from .simulate import simulate

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def main():
    # 1. set up parameters (you can tweak later)
    p = Params(N=50, T=20, seeds_f0=5, seeds_r0=5,gamma_correction=0.7, gamma_switch=0.3)
    out = simulate(p)
    
    # 2. run the simulation
    out = simulate(p)

    # 3. attach metadata (useful for reproducibility)
    out = {
        **out,
        "meta": {
            "params": p.__dict__,
            "created_at": _timestamp()
        }
    }

    # 4. save to JSON file
    os.makedirs("data/runs", exist_ok=True)
    filename = f"data/runs/CA_run_{_timestamp()}.json"
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {filename}")

    # 5. print friendly summary
    print("Results:")
    print(f"Peak fake posters: {max(out['I_f'])}, Peak real posters: {max(out['I_r'])}")
    print(f"Total shares — Fake: {out['cum_shares_f']}, Real: {out['cum_shares_r']}")
    print(f"Total switches (fake→real): {sum(out['switches'])}")
    print(f"Reach — Fake: {out['reach_fake']:.2%}, Real: {out['reach_real']:.2%}")

if __name__ == "__main__":
    main()
