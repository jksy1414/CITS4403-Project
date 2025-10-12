# src/ca/run.py
from .states import Params
from .simulate import simulate
from utils.io import save_json
from utils.io_paths import timestamped_run_path

def main():
    p = Params(
        N=50, T=60,
        seeds_f0=6, seeds_r0=5,
        beta_see=0.35, beta_share_f=0.50, beta_share_r=0.40,
        gamma_correction=0.7, gamma_switch=0.3,
        delta_decay_f=0.12, delta_decay_r=0.06,
        rng_seed=None,  # set an int like 123 for exact reproducibility
    )
    out = simulate(p)
    path = timestamped_run_path(model_name="ca", prefix="CA_run")
    save_json(out, path)

    print(f"Results saved → {path}")
    print("Results:")
    print(f"Seed used: {out.get('seed_used')}")
    peak_f = max(out["I_f"]) if out["I_f"] else 0
    peak_r = max(out["I_r"]) if out["I_r"] else 0
    print(f"Peak fake (posters): {peak_f}")
    print(f"Peak real (posters): {peak_r}")
    print(f"Reach — Fake: {out['reach_fake']:.2%}, Real: {out['reach_real']:.2%}")
    print(f"Total shares — Fake: {sum(out['shares_f'])}, Real: {sum(out['shares_r'])}")
    print(f"Time to peak — Fake: {out['I_f'].index(peak_f) if peak_f else None}, Real: {out['I_r'].index(peak_r) if peak_r else None}")

if __name__ == "__main__":
    main()
