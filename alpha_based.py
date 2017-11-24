from cliffwalker import *
from constants import *
import numpy as np
import copy


def converged(Q, Q_):
    return np.linalg.norm(Q-Q_)/Q.size < 0.001


def alpha_to_ix(alpha, nb_atoms):
    return int(np.clip(alpha * nb_atoms, 0, nb_atoms))


def compute_xi(Q, s, a, s_):
    nb_atoms = Q.shape[-1]

    alpha = 1./nb_atoms

    # xi =


class CVarState:

    # TODO: log time indexing

    def __init__(self, nb_atoms, spacing):

        self.atoms = np.array([0] + [1. / spacing ** (nb_atoms - 1 - i) for i in range(nb_atoms)])

        self.values = np.zeros_like(self.atoms)  # TODO: fill

    def __getitem__(self, alpha):

        for ix, y in enumerate(self.atoms):
            if alpha == y:
                return self.values[ix]
            elif alpha > y:
                ddalpha = (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])
                return self.values[ix] + ddalpha * (alpha - self.atoms[ix-1])

    def ddalpha(self, alpha):

        for ix, y in enumerate(self.atoms):
            if alpha == y:
                raise NotImplementedError('mix?')
            elif alpha > y:
                return (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])

    def __setitem__(self, alpha, value):

        for ix, y in enumerate(self.atoms):
            if alpha == y:
                self.atoms[ix] =
            elif alpha > y:
                return self.atoms[ix] + (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])

    def var(self, alpha):
        pass

    def inverse_var(self, var):
        pass




class CVaRValue:

    def __init__(self, world, nb_atoms):

        self.V = np.zeros((world.height, world.width, nb_atoms))

        self.spacing_const = 2
        self.eps = 0.1

        self.atoms = np.array([1./self.spacing_const**i for i in range(nb_atoms)]+[0])

    def __getitem__(self, item):
        y, x, alpha = item

        ix = int(np.clip(alpha * self.nb_atoms, 0, self.nb_atoms))
        r = alpha * self.nb_atoms - ix
        l = 1 - r
        return self.V[y, x, ix] * l + self.V[y, x, np.clip(ix + 1, 0, self.nb_atoms)] * r

    def __setitem__(self, key, value):
        y, x, alpha = key
        ix = int(np.round(alpha * self.nb_atoms))
        self.V[y, x, ix] = value

    def xi(self, s, t, alpha):
        pass


def value_update(world, V):
    """
    One value update step.
    """
    V_ = copy.deepcopy(V)
    for s in world.states():
        cvars = np.zeros((len(world.ACTIONS), V.nb_atoms))
        for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):

            t_p = np.array([t.prob for t in action_transitions])
            for ix, alpha in enumerate(np.arange(0, 1+1./V.nb_atoms, 1./V.nb_atoms)):

                t_q = np.zeros_like(t_p)

                for t_ix, t in enumerate(action_transitions):
                    xi = V.xi(s, t, alpha)

                    t_q[t_ix] = t.reward + gamma * xi * V[t.state.y, t.state.x, alpha*xi]

                cvars[a, ix] = np.dot(t_p, t_q)

    return V_


def value_iteration(world, nb_atoms):

    V = np.zeros((world.height, world.width, nb_atoms))

    i = 0
    while True:
        V_ = value_update(world, V)
        if converged(V, V_) and i != 0:
            print("value fully learned after %d iterations" % (i,))
            break
        V = V_
        i += 1
    return V



if __name__ == '__main__':

    # world = GridWorld(4, 6, random_action_p=0.1)

    print(np.array([1. / 2 ** i for i in range(21)] + [0]))






