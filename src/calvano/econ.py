import itertools
from dataclasses import dataclass

import numpy as np

# fsolve finds where the Nash first-order conditions equal zero
# minimize finds the minimum of minus joint profit, for monopoly
from scipy.optimize import fsolve, minimize


def demand(p, a, a0, mu):
    """Logit demand for n products plus an outside good. (Eq. 5)"""
    p, a = np.asarray(p, float), np.asarray(a, float)
    u = np.append((a - p) / mu, a0 / mu) # Last entry = outside good
    e = np.exp(u - u.max()) # Avoid overflow when mu is small
    return e[:-1] / e.sum()


def profit(p, a, a0, mu, c):
    """The per-period reward accruing to firm i (section II.B)."""
    return (np.asarray(p, float) - np.asarray(c, float)) * demand(p, a, a0, mu)


def nash_prices(a, a0, mu, c):
    """Return Bertrand-Nash prices under logit demand.
    
    Solves all firms' first order conditions p_i - c_i = mu / (1 - q_i)
    at once with fsolve, and raises an error if the solution isn't exact."""

    a = np.asarray(a, float)
    # Lets c be a single number when all firms share the same cost
    c = np.broadcast_to(np.asarray(c, float), a.shape).astype(float)
    
    # Each entry is zero when that firms' markup is optimal given rivals' prices
    foc = lambda p: p - c - mu / (1 - demand(p, a, a0, mu)) 

    # The optimal markup is always at least mu
    p = fsolve(foc, x0=c + mu) 
    
    return p


def monopoly_prices(a, a0, mu, c):
    """Returns the prices that maximise join profit (the collusive benchmark).
    
    Searches directly over the price vector with Nelder-Mead, using tight
    tolerances because pi^m enters every profit gain."""

    a = np.asarray(a, float)
    c = np.broadcast_to(np.asarray(c, float), a.shape).astype(float)

    # SciPy only minimises, so minimise mimus total industry profit
    obj = lambda p: -profit(p, a, a0, mu, c).sum() 

    # Monopoly prices are above Nash, so start above cost + mu
    res = minimize(obj, x0=c + 2 * mu, method='Nelder-Mead', 
                   options = dict(xatol=1e-12, fatol=1e-14, maxiter=20000))

    if not res.success:
        raise RuntimeError(f"Monopoly solver failed: {res.message}")

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
    
    With one-period memory, the state is the last period's joint action, written
    as a base-m number, with the last firm varying fastest."""
    return sum(a * m ** (len(actions) - 1 - i) for i, a in enumerate(actions))


def decode(s, m, n):
    """Convert a state number back into a list of price indices.
    
    Used to read learned strategies and price paths back as prices."""
    return [s // m ** (n - 1 - i) % m for i in range(n)]


def profit_tensor(grid, a, a0, mu, c):
    """Return every firm's profit for every combination of prices in the grid.
    
    Column k holds the profits when firms play the joint action that encode() maps 
    to k, so Q-learning can look up the profit rather than recomputing them."""

    n = len(a)

    # product() loops over prices
    combos = itertools.product(grid, repeat=n)

    # One column per joint action, giving the shape (n, m ** n)
    return np.column_stack([profit(p, a, a0, mu, c) for p in combos])

@dataclass(frozen=True)
class Params:
    """Economic environment, using the paper's baseline (section II.E)."""

    a: tuple = (2.0, 2.0) # Product qualities, one per firm
    c: tuple = (1.0, 1.0) # Marginal costs, one per firm
    a0: float = 0.0 # Outside good
    mu: float = 0.25 # Differentiation
    delta: float = 0.95 # Discount factor
    m: int = 15 # Number of grid prices
    xi: float = 0.1 # Grid extension beyond Nash and monopoly

@dataclass(frozen=True)
class Env:
    """Everything computed once by Params and shared by every session."""

    params: Params
    pN: np.ndarray # Nash prices, per firm
    pM: np.ndarray # Monopoly prices, per firm
    piN: float # Nash profit, averaged across firms (Eq. 9 uses per-firm average profits)
    piM: float # Monopoly profit, averaged across firms
    grid: np.ndarray # The m feasible prices
    PI: np.ndarray # Profit tensor

    def profit_gain(self, pibar):
        """Eq. 9: 0 at the Nash profit, 1 at the monopoly profit."""
        return (pibar - self.piN) / (self.piM - self.piN) 