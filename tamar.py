from cliffwalker import *
from util import *
from policies import AlphaBasedPolicy, VarBasedPolicy, NaiveCvarPolicy, FixedPolicy, GreedyPolicy
from random_variable import RandomVariable, MIN_VALUE, MAX_VALUE
import numpy as np
import copy


np.random.seed(1337)
np.set_printoptions(3)


NB_ATOMS = 4
SP = 2

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


def compute_cvar_by_sort(transition_p, var_values, atoms):
    """
    Naive CVaR computation
    :param transition_p:
    :param var_values: (transitions, nb_atoms)
    :param atoms: e.g. [0, 0.25, 0.5, 1]
    :return:
    """
    atom_p = atoms[1:] - atoms[:-1]

    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) sort
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]

    p_sorted, var_sorted = further_split(p_sorted, var_sorted, atoms)

    # 2) compute yV for each atom

    yV = np.zeros(NB_ATOMS)
    for ix, atom in enumerate(atoms[1:]):
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
    for i in range(NB_ATOMS):
        ddalpha = (yV[i]-last)/atom_p[i]
        var_solution[i] = ddalpha
        last = yV[i]

    return var_solution, yV


def init(world, construct):
    V = np.empty((world.height, world.width), dtype=object)
    for ix in np.ndindex(V.shape):

        V[ix] = construct()
    return V


def log_spaced_atoms(nb_atoms):
    return np.array([0] + [1. / SP ** (nb_atoms - 1 - i) for i in range(nb_atoms)])


class MarkovState:

    def __init__(self):
        self.var = np.zeros(NB_ATOMS)
        self.atoms = log_spaced_atoms(NB_ATOMS)

    def update(self, info):
        # action x (transition_probs, var)

        vars = []
        cvars = []
        for t_p, var in info:
            v, yv = compute_cvar_by_sort(t_p, var, self.atoms)
            vars.append(v)
            cvars.append(yv)

        vars = np.array(vars)
        cvars = np.array(cvars)
        best_args = np.argmax(cvars, axis=0)

        self.var = np.array([vars[best_args[i], i] for i in range(len(self.var))])

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)

        # atom_p = atoms[1:] - atoms[:-1]

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        # ax[1].plot(self.atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')
        plt.show()


def value_update(world, V):

    V_ = init(world, MarkovState)
    for s in world.states():

        full_info = []

        for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):
            # transition probabilities
            t_p = np.array([t.prob for t in action_transitions])

            # Tvars for next states
            t_v = np.array([V[t.state.y, t.state.x].var * gamma + t.reward for t in action_transitions])

            full_info.append((t_p, t_v))

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


if __name__ == '__main__':

    # world = GridWorld(1, 2, random_action_p=0.1)
    world = GridWorld(4, 6, random_action_p=0.1)

    # generate_multinomial(world_ideal)

    # =============== PI setup
    # 1/(3^4.5*(7/9)^10.5) = 0.1
    alpha = 0.1
    V = value_iteration(world)
    V[3,0].plot()

    greedy_policy = GreedyPolicy(V)
    # naive_cvar_policy = NaiveCvarPolicy(V, alpha=alpha)
    # var_policy = VarBasedPolicy(V, alpha=alpha)

    # exhaustive_stats(world_ideal, 1e6, GreedyPolicy, NaiveCvarPolicy, VarBasedPolicy)

    # =============== PI stats
    nb_epochs = 100000
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
    # plot_machine = PlotMachine(world, V_exp)
    # policy = var_policy
    # # policy = greedy_policy
    # for i in range(100):
    #     S, A, R = epoch(world, policy, plot_machine=plot_machine)
    #     print('{}: {}'.format(i, np.sum(R)))
    #     policy.reset()

    # ============== other
    # V_cvar = eval_fixed_policy(np.argmax(V_cvar, axis=0))
    # interesting = V_cvar[2, -1, 0]
    # print(interesting.cvar(alpha))
    # interesting.plot()



