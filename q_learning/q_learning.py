from cliffwalker import *
from util.constants import *
from util import cvar_computation
import numpy as np
from plots.grid import InteractivePlotMachine


class ActionValueFunction:

    def __init__(self, world, atoms):
        self.world = world
        self.atoms = atoms
        self.atom_p = self.atoms[1:] - self.atoms[:-1]

        self.Q = np.empty((world.height, world.width, len(world.ACTIONS)), dtype=object)
        for ix in np.ndindex(self.Q.shape):
            self.Q[ix] = MarkovState(self.atoms)

    def update_safe(self, x, a, x_, r, beta, id=None):
        """ TD update that ensures yCVaR convexity. """
        V_x = self.joint_action_dist(x_)

        for v in V_x:
            for i, atom in enumerate(self.atoms[1:]):
                V = self.Q[x.y, x.x, a].V[i]
                yC = self.Q[x.y, x.x, a].yC[i]

                # learning rates
                lr_v = beta * self.atom_p[i]  # p mirrors magnitude (for log-spaced)
                lr_yc = beta * self.atom_p[i]
                # lr_yc = beta * atom_p[i] / atom  # /atom for using the same beta when estimating cvar (not yc)

                if self.Q[x.y, x.x, a].V[i] >= r + gamma*v:
                    update = lr_v*(1-1/atom)
                else:
                    update = lr_v

                # UPDATE VaR
                if i == 0:
                    self.Q[x.y, x.x, a].V[i] = min(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i+1])
                elif i == (len(self.atoms)-2):
                    self.Q[x.y, x.x, a].V[i] = max(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i-1])
                else:
                    self.Q[x.y, x.x, a].V[i] = min(max(self.Q[x.y, x.x, a].V[i] + update, self.Q[x.y, x.x, a].V[i-1]),
                                                   self.Q[x.y, x.x, a].V[i+1])

                # UPDATE CVaR
                yCn = (1 - lr_yc) * yC + lr_yc * (atom*V + min(0, r+gamma*v - V))
                if i == 0:
                    self.Q[x.y, x.x, a].yC[i] = yCn
                elif i == 1:
                    ddy = self.Q[x.y, x.x, a].yC[0] / self.atom_p[0]  # TODO: check
                    self.Q[x.y, x.x, a].yC[i] = max(yCn, self.Q[x.y, x.x, a].yC[i - 1] + ddy * self.atom_p[i])
                else:
                    ddy = (self.Q[x.y, x.x, a].yC[i-1] - self.Q[x.y, x.x, a].yC[i-2]) / self.atom_p[i-1] # TODO: check
                    self.Q[x.y, x.x, a].yC[i] = max(yCn, self.Q[x.y, x.x, a].yC[i-1] + ddy*self.atom_p[i])

    def update_naive(self, x, a, x_, r, beta, id=None):
        """ CVaR TD update. """
        V_x = self.joint_action_dist(x_)
        print('standard')
        # TODO: vectorize/cythonize
        for iv, v in enumerate(V_x):
            for i, atom in enumerate(self.atoms[1:]):
                V = self.Q[x.y, x.x, a].V[i]
                yC = self.Q[x.y, x.x, a].yC[i]

                # learning rates
                lr_v = beta * self.atom_p[iv]  # p mirrors magnitude (for log-spaced)
                # lr_yc = beta * atom_p[iv] / atom  # /atom for using the same beta when estimating cvar (not yc)
                lr_yc = beta * self.atom_p[iv]

                if self.Q[x.y, x.x, a].V[i] >= r + gamma * v:
                    update = lr_v * (1 - 1 / atom)
                else:
                    update = lr_v

                # UPDATE VaR
                self.Q[x.y, x.x, a].V[i] += update

                # UPDATE CVaR
                yCn = (1 - lr_yc) * yC + lr_yc * (atom*V + min(0, r+gamma*v - V))
                self.Q[x.y, x.x, a].yC[i] = yCn
                # print(atom*V + min(0, r+gamma*v - V), end=', ')
        print(self.Q[x.y, x.x, a].yC)
        quit()

    def update(self, x, a, x_, r, beta, id=None):
        """ CVaR TD update. """
        V_x = self.joint_action_dist(x_)

        V = np.array(self.Q[x.y, x.x, a].V)
        yC = np.array(self.Q[x.y, x.x, a].yC)

        lr_v = beta * self.atom_p[:, np.newaxis]
        lr_yc = beta * self.atom_p

        # row is a single atom update
        # shape=(n, n)
        indicator_mask = self.Q[x.y, x.x, a].V >= r + gamma * V_x[:, np.newaxis]

        v_update = lr_v - indicator_mask * (lr_v / self.atoms[1:])

        self.Q[x.y, x.x, a].V += np.sum(v_update, axis=0)

        yCn = self.atoms[1:] * V + np.clip(r + gamma * V_x[:, np.newaxis] - V, None, 0)
        self.Q[x.y, x.x, a].yC = (1 - beta) * yC + beta * np.average(yCn, axis=0, weights=lr_yc)

    def next_action_alpha(self, x, alpha):
        yc = [self.Q[x.y, x.x, a].yc_alpha(alpha) for a in self.world.ACTIONS]
        return np.argmax(yc)

    def next_action_s(self, x, s):
        """
        Select best action according to E[(Z-s)^-].
        If multiple 0's, use yCVaR_0.
        """
        return max(self.world.ACTIONS, key=lambda a: (self.Q[x.y, x.x, a].cvar_pre_s(s), self.Q[x.y, x.x, a].yC[0]))

    def joint_action_dist(self, x, return_yc=False):
        """
        Returns a distribution representing the value function at state x.
        Constructed by taking a supremum of yC over actions for each atom.
        """
        yc = [np.max([self.Q[x.y, x.x, a].yC[i] for a in self.world.ACTIONS]) for i in range(NB_ATOMS)]

        if return_yc:
            return yc
        else:
            return cvar_computation.yc_to_var(self.atoms, yc)

    def joint_action_dist_var(self, x):
        """
        Returns VaR estimates of the joint distribution.
        Constructed by taking a supremum of yC over actions for each atom.
        """
        info = [max([(self.Q[x.y, x.x, a].yC[i], self.Q[x.y, x.x, a].V[i]) for a in self.world.ACTIONS]) for i in range(NB_ATOMS)]

        return [ycv[1] for ycv in info]

    def var_alpha(self, x, a, alpha):
        # TODO: check
        i = 0
        for i in range(len(self.atoms)):
            if alpha < self.atoms[i]:
                break
        return self.Q[x.y, x.x, a].V[i-1]

    def optimal_path(self, alpha):
        """ Optimal deterministic path. """
        from policy_improvement.policies import VarBasedQPolicy, XiBasedQPolicy, NaiveQPolicy
        from util.runs import optimal_path
        policy = VarBasedQPolicy(self, alpha)
        # policy = XiBasedQPolicy(self, alpha)
        # policy = NaiveQPolicy(self, alpha)
        return optimal_path(self.world, policy)


def is_ordered(v):
    for i in range(1, len(v)):
        if v[i-1] - v[i] > 1e-6:
            return False
    return True


def is_convex(yc, atoms):
    assert not LOG
    return is_ordered(cvar_computation.yc_to_var(atoms, yc))


class MarkovState:

    def __init__(self, atoms):
        self.atoms = atoms
        self.V = np.zeros(NB_ATOMS)  # VaR estimate
        self.yC = np.zeros(NB_ATOMS)  # CVaR estimate

    def plot(self, show=True, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 3)

        # var
        ax[0].step(self.atoms, list(self.V) + [self.V[-1]], 'o-', where='post')

        # yC
        ax[1].plot(self.atoms, np.insert(self.yC, 0, 0), 'o-')

        # yC-> var
        v = self.dist_from_yc()
        ax[2].step(self.atoms, list(v) + [v[-1]], 'o-', where='post')
        if show:
            plt.show()

    def expected_value(self):
        return self.yC[-1]

    def yc_alpha(self, alpha):
        """ linear interpolation: yC(alpha)"""
        i = 0
        for i in range(1, len(self.atoms)):
            if alpha < self.atoms[i]:
                break
        alpha_portion = (alpha - self.atoms[i-1]) / (self.atoms[i] - self.atoms[i-1])
        if i == 1:  # between 0 and first atom
            return alpha_portion * self.yC[i-1]
        else:
            return self.yC[i-2] + alpha_portion * (self.yC[i-1] - self.yC[i-2])

    def var_alpha(self, alpha):
        """ VaR estimate of alpha. """
        # TODO: check
        last_v = self.V[0]
        for p, v in zip(self.atoms[1:], self.V):
            if p > alpha:
                break
            last_v = v
        return last_v

    def cvar_pre_s(self, s):
        """ E[(V-s)^-] + ys.

        Uses the actual VaR for th cutoff and yC->VaR for the expectation.
        """
        yc = 0

        for ix, v_yc in enumerate(self.dist_from_yc()):
            v = self.V[ix]
            p = self.atoms[ix+1] - self.atoms[ix]
            if v < s:
                yc += p * v_yc
            else:
                break

        return yc

    def dist_from_yc(self):
        return cvar_computation.yc_to_var(self.atoms, self.yC)


def q_learning(world, alpha, max_episodes=2e3, max_episode_length=2e2):
    Q = ActionValueFunction(world, spaced_atoms(NB_ATOMS, SPACING, LOG))

    # learning parameters
    eps = 0.4
    beta = .6

    e = 0
    while e < max_episodes:
        if e % 10 == 0:
            print("e:{}, beta:{}".format(e, beta))
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

            s = (s-r)/gamma
            x = x_

            i += 1
        e += 1

        # if x in world.goal_states and r == -1:
        #     print('(success)')

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
    import time
    import pickle
    np.random.seed(2)
    start = time.time()

    # ============================= new config
    run_alpha = 0.1
    world = GridWorld(10, 15, random_action_p=0.1)
    Q = q_learning(world, run_alpha, max_episodes=20000)
    print('time=', time.time() - start)

    pickle.dump((world, Q), open('../files/q.pkl', mode='wb'))

    # ============================= load
    world, Q = pickle.load(open('../files/q.pkl', 'rb'))

    # ============================= RUN
    print('ATOMS:', spaced_atoms(NB_ATOMS, SPACING, LOG))

    for alpha in np.arange(0.05, 1, 0.05):
        print(alpha)
        pm = InteractivePlotMachine(world, Q, alpha=alpha, action_value=True)
        pm.show()
