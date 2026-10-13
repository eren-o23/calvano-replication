# Looping over all combinations of parameters for the economic model
import itertools

# Turning a class into one that just holds values to one with a constructor
from dataclasses import dataclass

import numpy as np

# Fsolve finds where a set of equations is equal to zero (Nash first-order conditions)
# Minimize finds the minimum of a function (minus joint profit, for monopoly)
from scipy.optimize import fsolve, minimize

def demand(p, a, a0, mu):
    """Logit demand for n products plus an outside good."""
    p, a = np.asarray(p, float), np.asarray(a, float)
    u = np.append((a - p) / mu, a0 / mu) # Last entry = outside good
    e = np.exp(u - u.max()) # Avoid overflow when mu is small
    return e[:-1] / e.sum()

def profit(p, a, a0, mu, c):
    return (np.asarray(p, float) - np.asarray(c, float)) * demand(p, a, a0, mu)

def nash_prices(a, a0, mu, c):
    """Return Bertrand-Nash prices under logit demand.
    
    Solves all firms' first order conditions p_i - c_i = mu / (1 - q_i)
    at once with fsolve, and raises an error if the solution isn't exact."""

    c = np.asarray(c, float)
    
    # Each entry is zero when that firms' markup is optimal given rivals' prices
    foc = lambda p: p - c - mu / (1 - demand(p, a, a0, mu)) 

    # The optimal markup is always at least mu
    p = fsolve(foc, x0=c + mu) 
    
    # Fsolve fails silently, and a wrong p^N would corrupt every profit gain later
    assert np.max(np.abs(foc(p))) < 1e-10, "Nash solver did not converge"
    
    return p