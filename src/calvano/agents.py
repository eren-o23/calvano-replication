import numpy as np
from numba import njit
from dataclasses import dataclass

from calvano.econ import encode_loop, decode

_encode = njit(cache=True)(encode_loop)


def initial_Q(env):
    """Return every agents starting Q-matrix, shape (n, m**n, m) (Eq. 8).
    
    Each cell is the profit agent i expects from price a_i if it's rivals
    pick prices uniformly at random, discounted forever. The same for every
    state: agents start with no idea that history matters, and learn it
    through play."""

    p = env.params
    n, m = p.n, p.m

    Q0 = np.empty((n, m**n, m))
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


@njit(cache=True)
def learn(PI, Q, alpha, beta, delta, seed, max_t, conv_window):
    """Run one session of Q-learning, updating Q in place.
    
    alpha and beta are arrays with one entry per agent, so agents can differ 
    in how fast they learn and explore. Returns (t_end, converged, last_state),
    where last_state is the joint action of the final period."""

    np.random.seed(seed)

    n, S, m = Q.shape

    # Reference table of each agent's preferred price in every state.
    strategy = np.empty((n, S), np.int64)
    for i in range(n):
        for s in range(S):
            strategy[i, s] = np.argmax(Q[i, s])

    s = np.random.randint(S) # Initial state drawn at random (Section II.E)
    a = np.empty(n, np.int64) # Allocated once, reused every period
    stable = 0 # Period since any agent's strategy changed

    for t in range(max_t):

        # Each agent explores with probability exp(-beta t) (Eq. 7).
        for i in range(n):
            if np.random.random() < np.exp(-beta[i] * t):
                a[i] = np.random.randint(m)
            else:
                a[i] = strategy[i, s]

        # The joint action is both the column of rewards and the next state.
        s_next = _encode(a, m)

        # Each agent updates the cell it just used (Eq. 4).
        changed = False
        for i in range(n):
            target = PI[i, s_next] + delta * Q[i, s_next].max()
            Q[i, s, a[i]] = (1 - alpha[i]) * Q[i, s, a[i]] + alpha[i] * target

            # Did the price change?
            best = np.argmax(Q[i, s])
            if best != strategy[i, s]:
                strategy[i, s] = best
                changed = True

        # Has the agent's strategy changed for conv_window periods (Section III.B)
        if changed:
            stable = 0
        else:
            stable += 1
            if stable >= conv_window:
                return t + 1, True, s_next

        s = s_next

    return max_t, False, s


def run_session(env, alpha, beta, seed, max_t=10**9, conv_window=100000):
    """Run one session from Q0 and return (Q, t_end, converged, last_state)."""

    n = env.params.n

    alpha = np.broadcast_to(np.asarray(alpha, float), (n,)).copy()
    beta = np.broadcast_to(np.asarray(beta, float), (n,)).copy()

    Q = initial_Q(env)
    t_end, converged, last_state = learn(env.PI, Q, alpha, beta, env.params.delta,
                                         seed, max_t, conv_window)
    return Q, t_end, converged, last_state


@dataclass(frozen=True)
class Outcome:
    """What a finished section settled on."""

    cycle: tuple # States in the limit cycle, in the order they're played
    prices: np.ndarray # Grid prices in each cycle state
    pibar: float # Average per-firm profit over the cycle
    gain: float # Profit gain (Eq. 9)
    strategy: np.ndarray # Greedy price index per agent and state

    @property
    def length(self):
        return len(self.cycle)

    @property
    def symmetric(self):
        # Every firm charges the same price in every state.
        return bool(np.all(self.prices == self.prices[:, :1]))


def limit_cycle(strategy, s0, m):
    """Follow the greedy strategies from state s0 until a state repeats.
    
    Returns the repeating part only, in the order that its played."""

    n = strategy.shape[0]
    first_visit = {} # Position in the path where it was first seen
    path = []
    s = int(s0)
    while s not in first_visit:
        first_visit[s] = len(path)
        path.append(s)
        # With no exploration, each agent plays its strategy, and those 
        # prices are the next state.
        s = int(encode_loop([strategy[i, s] for i in range(n)], m))
    return path[first_visit[s]:]


def session_outcome(env, Q, last_state):
    """Find the limit cycle a session ended on and its last profit gain."""

    p = env.params
    strategy = Q.argmax(axis=2)

    # Start from where the session really ended.
    cycle = limit_cycle(strategy, last_state, p.m)

    prices = env.grid[[decode(s, p.m, p.n) for s in cycle]]

    # Average over the cycle states for per firm profit.
    pibar = env.PI[:, cycle].mean()

    return Outcome(cycle=tuple(cycle), prices=prices, pibar=pibar,
                   gain=env.profit_gain(pibar), strategy=strategy)


def true_Q(env, strategy, i):
    """Agent i's true Q-matrix when both agents follow their learned strategies (Section III.D).
    
    Q(s, a) is the discounted profit from playing a once in state s, then both agents following
    their learned strategies forever (Eq. 3)."""

    p = env.params

    # Setting up the states.
    m, S = p.m, p.m ** p.n
    states = np.arange(S)

    # Own is every price agent i can try, rival is the other agent's learned price in each state.
    own, rival = np.arange(m), strategy[1 - i]

    # The core table, nxt[s, a] is the joint action played when agent i picks a in state s and 
    # the rival plays its strategy.
    if i == 0:
        nxt = own[None, :] * m + rival[:, None] # (S, m)
    else:
        nxt = rival[:, None] * m + own[None, :]

    # Agent i's profit for every state, price pair.
    reward = env.PI[i, nxt]

    # Both agents follow their strategies.
    follow = nxt[states, strategy[i]]

    # W(s) is the value of both agents following their strategies from s.
    # W = r + delta  * P W, where P moves each state to the next one from the strategies path.
    P = np.zeros((S, S))
    P[states, follow] = 1
    W = np.linalg.solve(np.eye(S) - p.delta * P, reward[states, strategy[i]])

    # Deviate to a for one period, then follow the strategies.
    return reward + p.delta * W[nxt]


def equilibrium_check(env, outcome, tol=1e-9):
    """Nash play on path and Q-loss, the same as table 1.
    
    Nash on path: in every cycle state, both agents learned prices are the best response.
    Q-loss: the share of the best attainable payoff an agent forfeits by its learned price."""

    S = env.params.m ** env.params.n
    path = np.array(outcome.cycle)

    # Nash becomes false if either agent fails
    # Both lists collect each agent's average Q-loss
    nash, loss_path, loss_all = True, [], []

    # Checks each agent in turn, getting its true Q-matrix against the other's frozen
    # strategy.
    for i in range(env.params.n):
        Q = true_Q(env, outcome.strategy, i)
        best = Q.max(axis=1)

        # Value of the price the agent learned in each state.
        played = Q[np.arange(S), outcome.strategy[i]]

        loss = (best - played) / best # Q-loss

        # Keeps nash true if the agent lost essentially nothing in every state.
        nash &= bool(np.all(loss[path] <= tol))

        loss_path.append(loss[path].mean())
        loss_all.append(loss.mean())

    return dict(nash=nash, qloss_path=float(np.mean(loss_path)),
                qloss_all=float(np.mean(loss_all)))

