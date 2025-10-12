from src.ca.states import Params
from src.ca.simulate import simulate

def test_reach_bounds():
    r = simulate(Params(T=20, rng_seed=42))
    assert 0.0 <= r["reach_fake"] <= 1.0
    assert 0.0 <= r["reach_real"] <= 1.0
