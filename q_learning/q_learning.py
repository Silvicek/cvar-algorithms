from cliffwalker import *
from util.constants import gamma
from util.util import spaced_atoms
from util import cvar_computation
import numpy as np

# atom spacing
NB_ATOMS = 4
LOG = False  # atoms are log-spaced
SPACING = 2

atoms = spaced_atoms(NB_ATOMS, SPACING, LOG)    # e.g. [0, 0.25, 0.5, 1]
atom_p = atoms[1:] - atoms[:-1]  # [0.25, 0.25, 0.5]

# learning parameters
eps = 0.1
beta = 0.1


class ActionValueFunction:

    def __init__(self, world):
        self.world = world

        self.Q = np.empty((world.height, world.width, len(world.ACTIONS)), dtype=object)
        for ix in np.ndindex(self.Q.shape):
            self.Q[ix] = MarkovState()

    def update(self, x, a, x_, r):

        # 'sampling'
        V_x = self.sup_q(x_)

        # TODO: deal with 0, 1
        for i, atom in enumerate(atoms[1:]):
            for v in V_x:
                V = self.Q[x.y, x.x, a].V[i]
                yC = self.Q[x.y, x.x, a].yC[i]

                if self.Q[x.y, x.x, a].V[i] <= r + gamma*v:
                    self.Q[x.y, x.x, a].V[i] -= beta*(1-1/atom)
                    self.Q[x.y, x.x, a].yC[i] = (1-beta) * yC + beta * (atom*V + r + gamma*v - V)
                else:
                    self.Q[x.y, x.x, a].V[i] -= beta
                    self.Q[x.y, x.x, a].yC[i] = (1 - beta) * yC + beta * (atom * V)

    def next_action_alpha(self, x, alpha):
        yc = [self.Q[x.y, x.x, a].yc_alpha(alpha) for a in self.world.ACTIONS]
        return np.argmax(yc)

    def next_action_s(self, x, s):
        yc = [self.Q[x.y, x.x, a].e_min_s(s) for a in self.world.ACTIONS]
        return np.argmax(yc)

    def sup_q(self, x):
        """
        Returns a distribution representing the value function at state x.
        Constructed by taking a supremum of yC over actions for each atom.
        """
        yc = [np.max([self.Q[x.y, x.x, a].yC[i] for a in self.world.ACTIONS]) for i in range(NB_ATOMS)]

        return cvar_computation.yc_to_var(atoms, yc)

    def var_alpha(self, x, a, alpha):
        # TODO: check
        i = 0
        for i in range(len(atoms)):
            if alpha > atoms[i]:
                break
        return self.Q[x.x, x.y, a].V[i-1]


class MarkovState:

    def __init__(self):
        self.V = np.zeros(NB_ATOMS)
        self.yC = np.zeros(NB_ATOMS)

    def plot(self, show=True):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        # ax[1].plot(self.atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')
        if show:
            plt.show()

    def expected_value(self):
        var = self.dist_from_yc()
        return np.dot(atom_p, var)

    def yc_alpha(self, alpha):
        # TODO: check
        i = 0
        for i in range(len(atoms)):
            if alpha > atoms[i]:
                break
        alpha_portion = (alpha - atoms[i-1]) / (atoms[i] - atoms[i-1])

        return self.yC[i-1] + alpha_portion * (self.yC[i] - self.yC[i-1])

    def e_min_s(self, s):
        # TODO: check
        e_min = 0
        for p, v in zip(atom_p, self.V):
            if v < s:
                e_min += p * v
            else:
                break
        return e_min

    def dist_from_yc(self):
        return cvar_computation.yc_to_var(atoms, self.yC)


def q_learning(world, alpha, max_episodes=1e3, max_episode_length=1e3):
    Q = ActionValueFunction(world)

    e = 0
    while e < max_episodes:
        x = world.initial_state
        a = Q.next_action_alpha(x, alpha)
        s = Q.var_alpha(x, a, alpha)
        i = 0
        while x not in world.goal_states and i < max_episodes:
            a = Q.next_action_s(x, s)
            x_, r = world.sample_transition(x, a)
            Q.update(x, a, x_, r)

            s = (s-r)/gamma
            x = x_

            i += 1
        e += 1

    return Q


if __name__ == '__main__':

    # world = GridWorld(1, 3, random_action_p=0.3)
    world = GridWorld(4, 6, random_action_p=0.1)

    print('ATOMS:', spaced_atoms(NB_ATOMS, SPACING, LOG))

    # =============== PI setup
    alpha = 0.1
    Q = q_learning(world, alpha)
    # V.V[3,0].plot()
    # print(V.V[1,5].var)
    # print(V.V[3,0].y_cvar(1.0))
    # print(V.V[3,0].expected_value())




