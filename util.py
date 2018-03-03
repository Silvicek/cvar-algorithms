from constants import *
import numpy as np


# gives a state-value function v(s) based on action-value function q(s, a) and policy
# for debugging and visualization
def q_to_v_argmax(world, Q):
    Vnew = np.zeros((world.height, world.width))
    for s in world.states():
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


def spaced_atoms(nb_atoms):
    if LOG:
        if SPACING < 2:
            return np.array([0, 0.5 / SPACING ** (nb_atoms-2)] + [1. / SPACING ** (nb_atoms - 1 - i) for i in range(1, nb_atoms)])
        return np.array([0] + [1. / SPACING ** (nb_atoms - 1 - i) for i in range(nb_atoms)])
    else:
        return np.linspace(0, 1, nb_atoms+1)


def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)

