from src.ca.states import Params
from src.ca.simulate import simulate

def test_misclass_runs_and_bounds():
    r = simulate(Params(T=10, rng_seed=7, micro_misclass=True, eta_misclass=0.05))
    assert 0.0 <= r["reach_fake"] <= 1.0
    assert 0.0 <= r["reach_real"] <= 1.0
    assert len(r["I_f"]) == 10 and len(r["I_r"]) == 10

def test_eta_zero_matches_baseline():
    p0 = Params(T=30, rng_seed=123, micro_misclass=False, eta_misclass=0.0)
    p1 = Params(T=30, rng_seed=123, micro_misclass=True,  eta_misclass=0.0)
    r0 = simulate(p0)
    r1 = simulate(p1)
    # With η=0, enabling the toggle should not change dynamics
    assert r0["I_f"] == r1["I_f"]
    assert r0["I_r"] == r1["I_r"]
