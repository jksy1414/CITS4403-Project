# tests/test_refractory.py
from src.ca.states import Params
from src.ca.simulate import simulate

def test_refractory_runs_and_bounds():
    r = simulate(Params(T=10, rng_seed=42, micro_refractory=True, tau_post=3))
    assert 0.0 <= r["reach_fake"] <= 1.0
    assert 0.0 <= r["reach_real"] <= 1.0
    assert len(r["I_f"]) == 10 and len(r["I_r"]) == 10

def test_refractory_changes_dynamics():
    p0 = Params(T=30, rng_seed=123, micro_refractory=False)
    p1 = Params(T=30, rng_seed=123, micro_refractory=True, tau_post=3)
    r0 = simulate(p0)
    r1 = simulate(p1)
    # typically changes peaks or timing
    assert (max(r0["I_f"]) != max(r1["I_f"])) or (max(r0["I_r"]) != max(r1["I_r"]))
