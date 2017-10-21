from cliffwalker import *
from constants import *
import numpy as np


# gives a state-value function v(s) based on action-value function q(s, a) and policy
# for debugging and visualization
def q_to_v_argmax(Q):
    Vnew = np.zeros((H, W))
    for s in states():
        a = np.argmax(Q[:, s.y, s.x])
        Vnew[s.y, s.x] = Q[a, s.y, s.x]
    return Vnew


def expected_value(rv):
    return rv.expected_value()


def cvar(rv, alpha):
    return rv.cvar(alpha)

expected_value = np.vectorize(expected_value)
cvar = np.vectorize(cvar)


def clip(ix):
    new_ix = max(0, min(MAX_VALUE - MIN_VALUE, ix))
    return new_ix

