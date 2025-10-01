import os, json, time
from .states import Params
from .simulate import simulate

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def main():
    # 1. set up parameters (you can tweak later)
    p = Params(N=50, T=20, seeds_f0=5, seeds_r0=2)
    
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
    peak_f = max(out["I_f"]) if out["I_f"] else 0
    peak_r = max(out["I_r"]) if out["I_r"] else 0
    print(f"Peak fake posters: {peak_f}")
    print(f"Peak real posters: {peak_r}")

if __name__ == "__main__":
    main()
