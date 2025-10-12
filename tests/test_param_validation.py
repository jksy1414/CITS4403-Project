import pytest
from src.ca.states import Params

def test_invalid_prob_raises():
    with pytest.raises(ValueError):
        Params(beta_see=1.2)

def test_too_many_seeds_raises():
    with pytest.raises(ValueError):
        Params(N=5, T=10, seeds_f0=30, seeds_r0=0)

def test_negative_seed_raises():
    with pytest.raises(ValueError):
        Params(seeds_f0=-1)
