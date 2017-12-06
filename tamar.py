from cliffwalker import *
from util import *
from policies import TamarPolicy
from visual import PlotMachine
from random_variable import RandomVariable, MIN_VALUE, MAX_VALUE
import numpy as np
import copy
import time
from pulp import *


np.random.seed(1337)
np.set_printoptions(3)


def further_split(p, v, atoms):
    cp = 0.
    atom_ix = 1
    new_p = []
    new_v = []

    for ix, (p_, v_) in enumerate(zip(p, v)):
        while abs(p_) > 1e-5:
            if cp+p_ >= atoms[atom_ix]:
                p__ = atoms[atom_ix] - cp
                p_ = p_ - p__
                atom_ix += 1
                cp += p__

                new_p.append(p__)
                new_v.append(v_)

            else:
                cp += p_
                new_p.append(p_)
                new_v.append(v_)
                p_ = 0

    return new_p, new_v


def init(world, construct):
    V = np.empty((world.height, world.width), dtype=object)
    for ix in np.ndindex(V.shape):

        V[ix] = construct()
    return V


def log_spaced_atoms(nb_atoms):
    spacing = 2
    return np.array([0] + [1. / spacing ** (nb_atoms - 1 - i) for i in range(nb_atoms)])


class MarkovState:

    def __init__(self, nb_atoms=5):
        self.nb_atoms = nb_atoms
        self.var = np.zeros(nb_atoms)
        self.atoms = log_spaced_atoms(nb_atoms)         # e.g. [0, 0.25, 0.5, 1]
        self.atom_p = self.atoms[1:] - self.atoms[:-1]  # [0.25, 0.25, 0.5]

    def update(self, info):
        # action x (transition, var)

        self.last_info = info

        vars = []
        cvars = []
        for t, var in info:
            v, yv = self.compute_cvar_by_sort([t_.prob for t_ in t], var)
            vars.append(v)
            cvars.append(yv)

        vars = np.array(vars)
        cvars = np.array(cvars)
        best_args = np.argmax(cvars, axis=0)

        self.var = np.array([vars[best_args[i], i] for i in range(len(self.var))])

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        # ax[1].plot(self.atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')
        plt.show()

    def cvar(self, alpha):
        atom_p = self.atoms[1:] - self.atoms[:-1]
        if alpha == 0:
            print('dont care anymore')
            return min(self.var)
        elif alpha == 1:
            print('expected value from now on')
            return np.dot(self.var, atom_p)

        p = 0.
        cv = 0.
        for p_, v_ in zip(atom_p, self.var):
            if p + p_ >= alpha:
                cv += (alpha-p) * v_
                break
            else:
                cv += p_ * v_
                p += p_

        return cv / alpha

    def compute_cvar_by_sort(self, transition_p, var_values):
        """
        CVaR computation from definition
        :param transition_p:
        :param var_values: (transitions, nb_atoms)
        :param atoms: e.g. [0, 0.25, 0.5, 1]
        :return:
        """
        # 0) weight by transition probs
        p = np.outer(transition_p, self.atom_p).flatten()

        # 1) sort
        sortargs = var_values.flatten().argsort()
        var_sorted = var_values.flatten()[sortargs]
        p_sorted = p.flatten()[sortargs]

        p_sorted, var_sorted = further_split(p_sorted, var_sorted, self.atoms)

        # 2) compute yV for each atom
        yV = np.zeros(self.nb_atoms)
        for ix, atom in enumerate(self.atoms[1:]):
            cs = 0.
            cp = 0.
            for p_, v_ in zip(p_sorted, var_sorted):
                cp += p_
                cs += p_ * v_
                if cp == atom:
                    break
            yV[ix] = cs

        # 3) get vars from yV
        last = 0.
        var_solution = np.zeros_like(yV)
        for i in range(self.nb_atoms):
            ddalpha = (yV[i] - last) / self.atom_p[i]
            var_solution[i] = ddalpha
            last = yV[i]

        return var_solution, yV

    def reconstruct_xis(self, transition_p, var_values, xi_alpha):
        # transform all to tuples (p, i_t, v)

        info = np.empty(var_values.shape, dtype=object)
        for i_t, p_t in enumerate(transition_p):
            for i_v, v, p_ in zip(range(len(var_values[i_t])), var_values[i_t], self.atom_p):
                info[i_t, i_v] = (p_*p_t, i_t, v)

        info = list(info.flatten())
        info.sort(key=lambda x: x[-1])

        xis = np.zeros_like(transition_p)
        p = 0.
        cv = 0.
        for p_, i, v in info:
            if p + p_ >= xi_alpha:
                xis[i] += alpha-p
                cv += (alpha-p) * v
                break
            else:
                xis[i] += p_
                cv += p_ * v

        return cv, xis / transition_p

    # def next_action_xis(self, alpha, transition=None):
    #
    #     best = (-1e6, 0, 0)
    #     a = 0
    #     for t, var in self.last_info:
    #         cv, xis = self.reconstruct_xis([t_.prob for t_ in t], var, alpha)
    #         if cv > best[0]:
    #             best = (cv, xis, a)
    #         a += 1
    #
    #     _, xis, a = best
    #
    #     if transition is None:
    #         return a, None
    #
    #     t_ix = self.last_info[a][0].index(transition)
    #
    #     return a, xis[t_ix]

    def next_action_xis(self, alpha, transition=None):

        if alpha == 0:

            return 3, 0


        best = (-1e6, 0, 0)
        a = 0
        for t, var in self.last_info:
            cv, xis = self.tamar_lp_single(a, alpha)
            if cv > best[0]:
                best = (cv, xis, a)
            a += 1

        _, xis, a = best

        if transition is None:
            return a, None

        t_ix = self.last_info[a][0].index(transition)

        return a, xis[t_ix]


    def tamar_lp_single(self, a, alpha):
        """
        Create LP:
        min Sum p_t * I

        0 <= xi <= 1/alpha
        Sum p_t * xi == 1

        I = max{yV}

        return yV[alpha]
        """
        if alpha == 0:
            return 0.
        nb_transitions = 4

        transition_p = [t.prob for t in self.last_info[a][0]]
        var_values = self.last_info[a][1]

        Xi = [LpVariable('xi_' + str(i)) for i in range(nb_transitions)]
        I = [LpVariable('I_' + str(i)) for i in range(nb_transitions)]

        prob = LpProblem(name='tamar')

        for xi in Xi:
            prob.addConstraint(0 <= xi)
            prob.addConstraint(xi <= 1. / alpha)
        prob.addConstraint(sum([xi * p for xi, p in zip(Xi, transition_p)]) == 1)

        for xi, i, var in zip(Xi, I, var_values):
            last = 0.
            f_params = []
            for ix in range(self.nb_atoms):
                k = var[ix]
                last += k * self.atom_p[ix]
                q = last - k * self.atoms[ix + 1]
                prob.addConstraint(i >= k * xi * alpha + q)
                f_params.append((k, q))

        # opt criterion
        prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

        prob.solve()

        return value(prob.objective), [value(xi)*alpha for xi in Xi]



def value_update(world, V):

    V_ = init(world, MarkovState)
    for s in world.states():

        full_info = []

        for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):
            # transition probabilities
            t_p = np.array([t.prob for t in action_transitions])

            # Tvars for next states
            t_v = np.array([V[t.state.y, t.state.x].var * gamma + t.reward for t in action_transitions])

            full_info.append((action_transitions, t_v))

        V_[s.y, s.x].update(full_info)

    return V_


def converged(V, V_, world):
    eps = 1e-3
    for s in world.states():
        dist = np.sum((V[s.y, s.x].var-V_[s.y, s.x].var)**2)
        if dist > eps:
            return False
    return True


def value_iteration(world):
    V = init(world, MarkovState)
    i = 0
    while True:
        # V[0,0].plot()
        V_ = value_update(world, V)
        print('UPDATED')
        # if i > 10:
        #     break
        if converged(V, V_, world) and i != 0:
            print("value fully learned after %d iterations" % (i,))
            break
        V = V_
        i += 1
    return V


def epoch(world, policy, max_iters=100, plot_machine=None):
    """
    Evaluates a single epoch starting at start_state, using a given policy.
    :param start_state:
    :param policy: Policy instance
    :param max_iters: end the epoch after this #steps
    :return: States, Actions, Rewards
    """
    s = world.initial_state
    S = [s]
    A = []
    R = []
    i = 0
    r = 0
    t = Transition(s, 0, 0)
    while s not in world.goal_states and i < max_iters:
        a = policy.next_action(t)

        if plot_machine is not None:
            plot_machine.step(s, a)
            time.sleep(0.5)


        A.append(a)
        trans = world.transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        t = trans[np.random.choice(len(trans), p=state_probs)]

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


if __name__ == '__main__':

    # world = GridWorld(1, 2, random_action_p=0.1)
    world = GridWorld(4, 6, random_action_p=0.3)

    # generate_multinomial(world_ideal)

    # =============== PI setup
    # 1/(3^4.5*(7/9)^10.5) = 0.1
    alpha = 0.1
    V = value_iteration(world)
    V[3,0].plot()

    tamar_policy = TamarPolicy(V, alpha)

    # greedy_policy = GreedyPolicy(V)
    # naive_cvar_policy = NaiveCvarPolicy(V, alpha=alpha)
    # var_policy = VarBasedPolicy(V, alpha=alpha)

    # exhaustive_stats(world_ideal, 1e6, GreedyPolicy, NaiveCvarPolicy, VarBasedPolicy)

    # =============== PI stats
    # nb_epochs = 100000
    # policy_stats(world_ideal, greedy_policy, alpha, nb_epochs=nb_epochs)
    # policy_stats(world_ideal, var_policy, alpha, nb_epochs=nb_epochs)

    # policy_stats(world_tweaked, greedy_policy, alpha, nb_epochs=nb_epochs)
    # policy_stats(world_tweaked, var_policy, alpha, nb_epochs=nb_epochs)

    # policy_stats(world_ideal, naive_cvar_policy, alpha, nb_epochs=nb_epochs)

    # =============== plot fixed
    # V_exp = expected_value(V)
    # V_exp = V_to_v_argmax(world, V_exp)
    # V_cvar = cvar(V, alpha)
    # V_cvar = V_to_v_argmax(world, V_cvar)
    # show_fixed(initial_state, V_to_v_argmax(V_exp), np.argmax(V_exp, axis=0))
    # show_fixed(initial_state, V_to_v_argmax(V_cvar), np.argmax(V_cvar, axis=0))

    # =============== plot dynamic
    V_visual = np.array([[V[i,j].cvar(alpha) for j in range(len(V[i]))]for i in range(len(V))])
    plot_machine = PlotMachine(world, V_visual)
    policy = tamar_policy
    # policy = greedy_policy
    for i in range(100):
        S, A, R = epoch(world, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()

    # ============== other
    # V_cvar = eval_fixed_policy(np.argmax(V_cvar, axis=0))
    # interesting = V_cvar[2, -1, 0]
    # print(interesting.cvar(alpha))
    # interesting.plot()



