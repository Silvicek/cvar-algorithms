from cliffwalker import *
from util.constants import *
from util import cvar_computation
import numpy as np
from plots.grid_plot_machine import InteractivePlotMachine


class ActionValueFunction:

    def __init__(self, world):
        self.world = world
        self.atoms = atoms

        self.Q = np.empty((world.height, world.width, len(world.ACTIONS)), dtype=object)
        for ix in np.ndindex(self.Q.shape):
            self.Q[ix] = MarkovState()

    def update_safe(self, x, a, x_, r, beta, id=None):
        # 'sampling'
        V_x = self.sup_q(x_)

        # TODO: deal with 0, 1
        for v in V_x:
            for i, atom in enumerate(atoms[1:]):
                V = self.Q[x.y, x.x, a].V[i]
                yC = self.Q[x.y, x.x, a].yC[i]

                # learning rates
                lr_v = beta * atom_p[i]  # p mirrors magnitude (for log-spaced)
                lr_yc = beta * atom_p[i]
                # lr_yc = beta * atom_p[i] / atom  # /atom for using the same beta when estimating cvar (not yc)

                if self.Q[x.y, x.x, a].V[i] >= r + gamma*v:
                    update = lr_v*(1-1/atom)
                else:
                    update = lr_v

                # UPDATE VaR
                if i == 0:
                    self.Q[x.y, x.x, a].V[i] = min(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i+1])
                elif i == (len(atoms)-2):
                    self.Q[x.y, x.x, a].V[i] = max(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i-1])
                else:
                    self.Q[x.y, x.x, a].V[i] = min(max(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i-1]),
                                                   self.Q[x.y, x.x, a].V[i+1])

                # UPDATE CVaR
                yCn = (1 - lr_yc) * yC + lr_yc * (atom*V + min(0, r+gamma*v - V))
                if i == 0:
                    self.Q[x.y, x.x, a].yC[i] = yCn
                elif i == 1:
                    ddy = self.Q[x.y, x.x, a].yC[0] / atom_p[0]  # TODO: check
                    self.Q[x.y, x.x, a].yC[i] = max(yCn, self.Q[x.y, x.x, a].yC[i - 1] + ddy * atom_p[i])
                else:
                    ddy = (self.Q[x.y, x.x, a].yC[i-1] - self.Q[x.y, x.x, a].yC[i-2]) / atom_p[i-1] # TODO: check
                    self.Q[x.y, x.x, a].yC[i] = max(yCn, self.Q[x.y, x.x, a].yC[i-1] + ddy*atom_p[i])

    def update(self, x, a, x_, r, beta, id=None):
        # 'sampling'
        V_x = self.sup_q(x_)

        # TODO: deal with 0, 1
        for iv, v in enumerate(V_x):
            for i, atom in enumerate(atoms[1:]):
                V = self.Q[x.y, x.x, a].V[i]
                yC = self.Q[x.y, x.x, a].yC[i]

                # learning rates
                lr_v = beta * atom_p[iv]  # p mirrors magnitude (for log-spaced)
                # lr_yc = beta * atom_p[iv] / atom  # /atom for using the same beta when estimating cvar (not yc)
                lr_yc = beta * atom_p[iv]

                if self.Q[x.y, x.x, a].V[i] >= r + gamma * v:
                    update = lr_v * (1 - 1 / atom)
                else:
                    update = lr_v

                # UPDATE VaR
                self.Q[x.y, x.x, a].V[i] += update

                # UPDATE CVaR
                yCn = (1 - lr_yc) * yC + lr_yc * (atom*V + min(0, r+gamma*v - V))
                self.Q[x.y, x.x, a].yC[i] = yCn

    def next_action_alpha(self, x, alpha):
        yc = [self.Q[x.y, x.x, a].yc_alpha(alpha) for a in self.world.ACTIONS]
        return np.argmax(yc)

    def next_action_s(self, x, s):
        yc = [self.Q[x.y, x.x, a].e_min_s(s) for a in self.world.ACTIONS]
        return np.argmax(yc)

    def sup_q(self, x, return_yc=False):
        """
        Returns a distribution representing the value function at state x.
        Constructed by taking a supremum of yC over actions for each atom.
        """
        yc = [np.max([self.Q[x.y, x.x, a].yC[i] for a in self.world.ACTIONS]) for i in range(NB_ATOMS)]

        if return_yc:
            return yc
        else:
            return cvar_computation.yc_to_var(atoms, yc)

    def var_alpha(self, x, a, alpha):
        # TODO: check
        i = 0
        for i in range(len(atoms)):
            if alpha < atoms[i]:
                break
        return self.Q[x.x, x.y, a].V[i-1]


def is_ordered(v):
    for i in range(1, len(v)):
        if v[i-1] - v[i] > 1e-6:
            return False
    return True


def is_convex(yc):
    assert not LOG
    return is_ordered(cvar_computation.yc_to_var(atoms, yc))


class MarkovState:

    def __init__(self):
        self.V = np.zeros(NB_ATOMS)
        self.yC = np.zeros(NB_ATOMS)

    def plot(self, show=True, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 3)

        # var
        ax[0].step(atoms, list(self.V) + [self.V[-1]], 'o-', where='post')

        # yC
        ax[1].plot(atoms, np.insert(self.yC, 0, 0), 'o-')

        # yC-> var
        v = self.dist_from_yc()
        ax[2].step(atoms, list(v) + [v[-1]], 'o-', where='post')
        if show:
            plt.show()

    def expected_value(self):
        return self.yC[-1]

    def yc_alpha(self, alpha):
        """ linear interpolation: yC(alpha)"""
        i = 0
        for i in range(1, len(atoms)):
            if alpha < atoms[i]:
                break
        alpha_portion = (alpha - atoms[i-1]) / (atoms[i] - atoms[i-1])
        if i == 1:  # between 0 and first atom
            return alpha_portion * self.yC[i-1]
        else:
            return self.yC[i-2] + alpha_portion * (self.yC[i-1] - self.yC[i-2])

    def e_min_s(self, s):
        """ E[(V-s)^-] """
        e_min = 0
        for p, v in zip(atom_p, self.V):
            if v < s:
                e_min += p * (v - s)
            else:
                break
        return e_min

    def dist_from_yc(self):
        return cvar_computation.yc_to_var(atoms, self.yC)


def q_learning(world, alpha, max_episodes=2e3, max_episode_length=1e2):
    Q = ActionValueFunction(world)

    # learning parameters
    eps = 0.5
    beta = .4

    e = 0
    while e < max_episodes:
        if e % 10 == 0:
            print(e, beta)
            beta *= 0.995
        x = world.initial_state
        a = eps_greedy(Q.next_action_alpha(x, alpha), eps, world.ACTIONS)
        s = Q.var_alpha(x, a, alpha)
        i = 0
        while x not in world.goal_states and i < max_episode_length:
            a = eps_greedy(Q.next_action_s(x, s), eps, world.ACTIONS)
            t = world.sample_transition(x, a)
            x_, r = t.state, t.reward

            Q.update(x, a, x_, r, beta, (e, i))
            # Q.update_safe(x, a, x_, r, beta, (e, i))

            s = (s-r)/gamma
            x = x_

            i += 1
        e += 1

    return Q


def pseudo_q_learning(world, max_episodes):
    Q = ActionValueFunction(world)

    e = 0
    beta = 0.01 / NB_ATOMS
    while e < max_episodes:
        if e % 10 == 0:
            print(e, beta)
        for x in world.states():
            if x in world.goal_states or x in world.cliff_states:
                continue
            a = np.random.randint(0, 4)

            t = world.sample_transition(x, a)
            x_, r = t.state, t.reward

            # Q.update(x, a, x_, r, beta)
            Q.update_safe(x, a, x_, r, beta)

        e += 1

    return Q


def eps_greedy(a, eps, action_space):
    if np.random.random() < eps:
        return np.random.choice(action_space)
    else:
        return a


def q_to_v_exp(Q):
    return np.max(np.array([Q.Q[ix].expected_value() for ix in np.ndindex(Q.Q.shape)]).reshape(Q.Q.shape), axis=-1)

if __name__ == '__main__':

    # world = GridWorld(1, 3, random_action_p=0.3)
    world = GridWorld(10, 15, random_action_p=0.1)

    print('ATOMS:', spaced_atoms(NB_ATOMS, SPACING, LOG))

    # =============== PI setup
    alpha = 0.1
    Q = q_learning(world, alpha, max_episodes=20000)
    # Q = pseudo_q_learning(world, 10000)
    import pickle
    pickle.dump(Q, open("../files/q.pkl", 'wb'))
    # Q = pickle.load(open("../files/q.pkl", 'rb'))

    pm = InteractivePlotMachine(world, Q, action_value=True)
    pm.show()



