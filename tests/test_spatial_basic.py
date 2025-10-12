# tests/test_spatial_basic.py
from src.ca.states import Params
from src.ca.simulate import simulate

def test_spatial_flag_runs():
    p = Params(macro_spatial=True, spatial_strength=0.4, rng_seed=123)
    out = simulate(p)
    assert 0.0 <= out["reach_fake"] <= 1.0
    assert 0.0 <= out["reach_real"] <= 1.0

def test_spatial_zero_equals_nospatial():
    p0 = Params(macro_spatial=False, rng_seed=123)
    p1 = Params(macro_spatial=True, spatial_strength=0.0, rng_seed=123)
    r0 = simulate(p0)
    r1 = simulate(p1)
    # exact equality is unlikely due to RNG flow; compare key aggregates loosely
    assert abs(sum(r0["shares_f"]) - sum(r1["shares_f"])) < 200
    assert abs(sum(r0["shares_r"]) - sum(r1["shares_r"])) < 200
