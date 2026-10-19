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

    a = np.asarray(a, float)
    # Broadcast to match a if c is a scalar
    c = np.broadcast_to(np.asarray(c, float), a.shape).astype(float)
    
    # Each entry is zero when that firms' markup is optimal given rivals' prices
    foc = lambda p: p - c - mu / (1 - demand(p, a, a0, mu)) 

    # The optimal markup is always at least mu
    p = fsolve(foc, x0=c + mu) 
    
    # Fsolve fails silently, and a wrong p^N would corrupt every profit gain later
    assert np.max(np.abs(foc(p))) < 1e-10, "Nash solver did not converge"
    
    return p

def monopoly_prices(a, a0, mu, c):
    """Returns the prices that maximise join profit (the collusive benchmark).
    
    Searches directly over the price vector with Nelder-Mead, using tight
    tolerances because pi^m enters every profit gain."""

    a = np.asarray(a, float)
    c = np.broadcast_to(np.asarray(c, float), a.shape).astype(float)

    # SciPy only minimises, so minimise mimus total industry profit
    obj = lambda p: -profit(p, a, a0, mu, c).sum() 

    # Monopoly prices start above Nash, so start above cost + mu
    res = minimize(obj, x0=c + 2 * mu, method='Nelder-Mead', 
                   options = dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    assert res.success, f"Monopoly solver failed: {res.message}"

    return res.x

def price_grid(pN, pM, m, xi):
    """Return m evenly spaced prices from just below Nash to just above monopoly.
    
    The range [pN, pM] is extended by xi times its width at each end
    (section II.B), so agents can price below Nash and above monopoly."""

    # One common grid for all firms; min / max only matters when firms are asymmetric
    lo, hi = pN.min(), pM.max()
    return np.linspace(lo - xi * (hi - lo), # Lower end: pN - ξ(pM - pN)
                       hi + xi * (hi - lo), # Upper end: pM + ξ(pM - pN)
                       m) # m equally spaced points

def encode(actions, m):
    """Convert a list of price indices into a single state number.
    
    Q-learning needs somewhere to store what each agent has learned."""
    return sum(a * m ** (len(actions) - 1 - i) for i, a in enumerate(actions))

def decode(s, m, n):
    """Convert a state number into a list of price indices.
    
    Hoping that this is useful for analysing the results."""
    return [s // m ** (n - 1 - i) % m for i in range(n)]

