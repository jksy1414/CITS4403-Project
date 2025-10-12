import numpy as np
from src.ca.states import Params
from src.ca.simulate import simulate

def test_hetero_noop_when_sd_zero():
    p0 = Params(T=20, rng_seed=42, macro_hetero=False, hetero_sd=0.0)
    p1 = Params(T=20, rng_seed=42, macro_hetero=True,  hetero_sd=0.0)
    r0 = simulate(p0)
    r1 = simulate(p1)
    # enabling hetero with sd=0 should not change dynamics
    assert r0["I_f"] == r1["I_f"]
    assert r0["I_r"] == r1["I_r"]

def test_hetero_fields_present():
    r = simulate(Params(T=10, rng_seed=7, macro_hetero=True, hetero_sd=0.2))
    m = r.get("macro", {})
    assert m.get("hetero") is True
    assert 0.0 <= m.get("f_mult_mean", 0.0) <= 2.0
    assert 0.0 <= m.get("r_mult_mean", 0.0) <= 2.0
    assert 0.0 <= m.get("g_mult_mean", 0.0) <= 2.0
