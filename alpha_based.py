# TODO: whats this for?
from cliffwalker import *
from constants import *
import numpy as np
import copy
import matplotlib.pyplot as plt


def converged(Q, Q_):
    return np.linalg.norm(Q-Q_)/Q.size < 0.001


class CVarState:

    # TODO: log time indexing (constant?)

    def __init__(self, nb_atoms, spacing):

        self.atoms = np.array([0] + [1. / spacing ** (nb_atoms - 2 - i) for i in range(nb_atoms-1)])
        print(self.atoms)

        self.values = np.zeros_like(self.atoms)  # TODO: better init?

    def __getitem__(self, alpha):
        for ix, y in enumerate(self.atoms):
            if alpha == y:
                return self.values[ix]
            elif alpha > y:
                ddalpha = (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])
                return self.values[ix] + ddalpha * (alpha - self.atoms[ix-1])

    def __setitem__(self, alpha, value):
        for ix, y in enumerate(self.atoms):
            if alpha == y:  # XXX: eps-close?
                self.atoms[ix] = value

    def ddalpha(self, alpha):
        for ix, y in enumerate(self.atoms):
            if alpha >= y:
                if alpha == y and ix == 0:
                    ix += 1

                return (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])

    def var(self, alpha):
        return self.ddalpha(alpha)

    def inverse_var(self, var):
        # TODO: deal with discreteness by linear decomposition (?)
        for ix, y in enumerate(self.atoms):
            if ix == 0:
                continue

            ddalpha = (self.values[ix] - self.values[ix - 1]) / (self.atoms[ix] - self.atoms[ix - 1])
            if ddalpha >= var:
                return y

    def plot(self):
        plt.semilogx(self.atoms, self.values)
        plt.show()


class CVaRValue:

    def __init__(self, world, nb_atoms):

        spacing_const = 2
        eps = 0.1

        self.V = np.empty((world.height, world.width), dtype=object)

        for y in range(world.height):
            for x in range(world.width):
                self.V[y, x] = CVarState(nb_atoms, spacing_const)

        self.world = world

        self.nb_atoms = nb_atoms

    def __getitem__(self, item):
        y, x, alpha = item
        return self.V[y, x][alpha]

    def __setitem__(self, key, value):
        y, x, alpha = key
        self.V[y, x][alpha] = value

    def interpolated(self, s, t, alpha):

        var = self.V[s.y, s.x].var(alpha)

        xi_alpha = self.V[t.state.y, t.state.x].inverse_var(var)

        return xi_alpha

    def value_update(self):
        """
        One value update step.
        """
        V_ = copy.deepcopy(self)
        for s in world.states():
            cvars = np.zeros((len(world.ACTIONS), self.nb_atoms))



            # for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):
            #
            #     t_p = np.array([t.prob for t in action_transitions])
            #     for ix, alpha in enumerate(self.V[s.y, s.x].atoms):
            #         if ix == 0:
            #             continue
            #
            #         t_q = np.zeros_like(t_p)
            #
            #         for t_ix, t in enumerate(action_transitions):
            #             new_cvar = self.interpolated(s, t, alpha)
            #
            #             t_q[t_ix] = t.reward + gamma * new_cvar
            #
            #         cvars[a, ix] = np.dot(t_p, t_q)

            for ix, alpha in enumerate(self.V[s.y, s.x].atoms):
                V_.V[s.y, s.x].values[ix] = np.max(cvars[:, ix])

        return V_

    def multiplot(self):

        fig, ax = plt.subplots(world.height, world.width, sharex=True, sharey=True)

        if world.height == 1:

                for x, ax_ in zip(range(world.width), ax):
                    # ax_.plot(self.V[0,x].atoms, self.V[0,x].values, 'o-')
                    ax_.step(self.V[0,x].atoms, self.V[0,x].values, 'o-')
        else:

            for y, ax_row in zip(range(world.height), ax):
                for x, ax_ in zip(range(world.width), ax_row):
                    ax_.semilogx(self.V[y,x].atoms, self.V[y,x].values)

        plt.show()


def value_iteration(world, nb_atoms):

    V = CVaRValue(world, nb_atoms)

    i = 0
    while True:
        V_ = V.value_update()

        V.multiplot()
        # if converged(V, V_) and i != 0:
        if i % 10 == 0 and i != 0:
            print("value fully learned after %d iterations" % (i,))
            break
        V = V_
        i += 1
    return V



if __name__ == '__main__':

    world = GridWorld(1, 2, random_action_p=0.1)

    V = value_iteration(world, 4)








