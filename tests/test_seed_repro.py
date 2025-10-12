from src.ca.states import Params
from src.ca.simulate import simulate

def test_seed_reproducibility():
    r1 = simulate(Params(rng_seed=123, T=20))
    r2 = simulate(Params(rng_seed=123, T=20))
    assert r1["I_f"] == r2["I_f"]
    assert r1["I_r"] == r2["I_r"]
