import numpy as np


def initial_Q(env):
    """Return every agents starting Q-matrix, shape (n, m**m, m) (Eq. 8).
    
    Each cell is the profit agent i expects from price a_i if it's rivals
    pick prices uniformly at random, discounted forever. The same for every
    state: agents start with no idea that history matters, and learn it
    through play."""

    p = env.params
    n, m = p.n, p.m

    Q0 = np.empty((n, m**m, m))
    for i in range(n):
        
        # One axis per firm
        PI_i = env.PI[i].reshape((m,)* n)

        # Bring agent i's own price to the front, then average over all
        # rivals' prices
        own = np.moveaxis(PI_i, i, 0).reshape(m, -1).mean(axis=1)

        # Divide by 1 - delta to value that profit every period forever,
        # then copy the same row into every state
        Q0[i] = own / (1 - p.delta)

    return Q0