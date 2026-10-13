# Looping over all combinations of parameters for the economic model
import itertools

# Turning a class into one that just holds values to one with a constructor
from dataclasses import dataclass

import numpy as np

# fsolve finds where a set of equations is equal to zero (Nash first-order conditions)
# minimize finds the minimum of a function (minus joint profit, for monopoly)
from scipy.optimize import fsolve, minimize

def demand(p, a, a0, mu):
    """Logit demand for n products plus an outside good."""
    p, a = np.asarray(p, float), np.asarray(a, float)
    u = np.append((a - p) / mu, a0 / mu) # last entry = outside good
    e = np.exp(u - u.max()) # avoid overflow when mu is small
    return e[:-1] / e.sum()

def profit(p, a, a0, mu, c):
    return (np.asarray(p, float) - np.asarray(c, float)) * demand(p, a, a0, mu)