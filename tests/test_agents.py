import numpy as np
import pytest

from calvano.econ import Params, build_env
from calvano.agents import initial_Q


@pytest.fixture(scope="module")
def env():
    return build_env()

@pytest.fixture(scope="module")
def Q0(env):
    return initial_Q(env)


# Initial Q-matrix (Eq. 8)


def test_Q0_shape(Q0):
    # One matrix per agent, one row per state (225), one column per price (15)
    assert Q0.shape(15, 15 ** 2, 15)

def test_Q0_uses_each_agents_own_axis():
    # With identical firms, a mix up between own and rivals axes would go unnoticed,
    # so agent 1 is given a different cost.
    env = build_env(Params(c=(1.0, 1.2)))
    PI = env.PI.reshape(2, 15, 15)
    Q0 = initial_Q(env)
    assert np.allclose(Q0[1, 0], PI[1 , :, :].mean(axis=0) / 0.05)
    assert np.allclose(Q0[0, 0], PI[0 , :, :].mean(axis=0) / 0.05)
