import numpy as np
import pytest

from calvano.econ import demand, profit, nash_prices, monopoly_prices, price_grid

# Baseline parameters
A, A0, MU, C = [2, 2], 0, 0.25, [1, 1]
M, XI = 15, 0.1

@pytest.fixture(scope="module")
def pN():
    return nash_prices(A, A0, MU, C)

@pytest.fixture(scope="module")
def pM():
    return monopoly_prices(A, A0, MU, C)

# Demand and profit (Eq. 5)


def test_demand_at_nash(pN):
    # Checks Eq. 5 evaluated at the Nash prices
    assert np.allclose(demand(pN, A, A0, MU), 0.4714, atol=1e-3)

def test_shares_sum_below_one(pN):
    # Checks the outside good is in the denominator of Eq. 5
    assert demand(pN, A, A0, MU).sum() < 1

def test_demand_falls_with_own_price():
    # Demand must slope down; raising price loses you customers
    # and they go to the rival
    q_low = demand([1.5, 1.5], A, A0, MU)
    q_high = demand([1.6, 1.5], A, A0, MU) # Firm 1 raises it's price
    assert q_high[0] < q_low[0] # Firm 1's share falls
    assert q_high[1] > q_low[1] # Firm 2's share rises

def test_no_overflow_with_small_mu():
    # Checks that demand still gives reasonable values when mu is very small
    q = demand([1.5, 1.6], A, A0, 0.001)
    assert np.all(np.isfinite(q))

def test_profit_at_nash(pN):
    # Reference value for the Nash benchmark
    assert np.allclose(profit(pN, A, A0, MU, C), 0.2229, atol=1e-3)


# Nash prices


def test_nash_prices(pN):
    # Compare both firms' prices to the papers reported Nash price
    assert np.allclose(pN, 1.47293, atol=1e-4)

def test_nash_foc_holds(pN):
    # Check that each firm's markup equals the optimal markup given its share,
    # so that no firm can gain by deviating. 
    q = demand(pN, A, A0, MU)
    assert np.allclose(pN - np.array(C), MU / (1 - q), atol=1e-8)




