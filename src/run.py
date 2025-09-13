import argparse, json, os
from .model import Params
from .simulate import simulate
from utils.plotting import plot_curves  # optional plotting

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="BA", choices=["ER","WS","BA"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--flag_prob", type=float, default=0.10)
    ap.add_argument("--plot", action="store_true", help="Show plots")
    args = ap.parse_args()

    p = Params(topology=args.topology, n=args.n, T=args.T, flag_prob=args.flag_prob)
    out = simulate(p)

    summary = {
        "N": int(out["N"][0]),
        "final_reach_F": int(out["ever_seen_F"][-1]),
        "final_reach_R": int(out["ever_seen_R"][-1]),
        "peak_F": int(out["posters_F"].max()),
        "peak_R": int(out["posters_R"].max()),
        "final_quarantined_F": int(out["quarant_F"][-1]),
    }
    print(json.dumps(summary, indent=2))

    os.makedirs("data", exist_ok=True)
    with open("data/last_run_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    if args.plot:
        plot_curves(out, f"{args.topology}")

if __name__ == "__main__":
    main()
