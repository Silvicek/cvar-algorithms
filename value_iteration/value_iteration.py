from cliffwalker import *
from util import *
import numpy as np
import copy
from pulp import *


def further_split(p, v, atoms):  # TODO: remove this
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


class ValueFunction:

    def __init__(self, world):
        self.world = world

        self.V = np.empty((world.height, world.width), dtype=object)
        for ix in np.ndindex(self.V.shape):
            self.V[ix] = MarkovState()

    def update(self, y, x):

        vars = []
        cvars = []

        for a in self.world.ACTIONS:
            t = list(self.transitions(y, x, a))

            var_values = self.transition_vars(y, x, a)

            if TAMAR_LP:
                v, yv = self.V[y, x].compute_cvar_by_lp([t_.prob for t_ in t], var_values)
            else:

                if WASSERSTEIN:
                    v, yv = self.V[y, x].compute_wasserstein([t_.prob for t_ in t], var_values)
                else:
                    v, yv = self.V[y, x].compute_cvar_by_sort([t_.prob for t_ in t], var_values)
            vars.append(v)
            cvars.append(yv)

        vars = np.array(vars)
        cvars = np.array(cvars)
        best_args = np.argmax(cvars, axis=0)

        self.V[y, x].var = np.array([vars[best_args[i], i] for i in range(len(self.V[y, x].var))])

    def next_action(self, y, x, alpha):
        assert alpha != 0

        best = (-1e6, 0, 0)
        for a in self.world.ACTIONS:

            if TAMAR_LP:
                cv, xis = self.tamar_lp_single(y, x, a, alpha)
            else:
                _, cv, xis = self.var_cvar_xis(y, x, a, alpha)

            if cv > best[0]:
                best = (cv, xis, a)

        _, xis, a = best

        return a, xis

    def tamar_lp_single(self, y, x, a, alpha):
        """
        Create LP:
        min Sum p_t * I

        0 <= xi <= 1/alpha
        Sum p_t * xi == 1

        I = max{yV}

        return yV[alpha]
        """

        transition_p = [t.prob for t in self.transitions(y, x, a)]
        var_values = self.transition_vars(y, x, a)
        nb_transitions = len(transition_p)

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
            for ix in range(self.V[y, x].nb_atoms):
                k = var[ix]
                last += k * self.V[y, x].atom_p[ix]
                q = last - k * self.V[y, x].atoms[ix + 1]
                prob.addConstraint(i >= k * xi * alpha + q)
                f_params.append((k, q))

        # opt criterion
        prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

        prob.solve()

        return value(prob.objective), [value(xi)*alpha for xi in Xi]

    def var_cvar_xis(self, y, x, a, alpha):
        """
        Compute VaR, CVaR and xi values in O(nlogn)
        :param y:
        :param x:
        :param a:
        :param alpha:
        :return: var, cvar, xis
        """
        # TODO: O(n)

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)

        info = extract_distribution(transitions, var_values, self.V[y, x].atom_p)

        xis = np.zeros(len(transitions))
        p = 0.
        cv = 0.
        v = 0.
        for p_, i, v in info:
            if p + p_ >= alpha:
                xis[i] += alpha-p
                cv += (alpha-p) * v
                break
            else:
                xis[i] += p_
                cv += p_ * v
                p += p_

        return v, cv, xis / np.array([t.prob for t in transitions])

    def y_var(self, y, x, a, var):

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)

        info = extract_distribution(transitions, var_values, self.V[y, x].atom_p)

        yv = 0.
        p = 0
        for p_, _, v_ in info:
            if v_ >= var:  # TODO: solve for discrete distributions
                break
            else:
                yv += p_ * v_
            p += p_

        return p, yv

    def transitions(self, y, x, a):
        for t in self.world.transitions(State(y, x))[a]:
            yield t

    def transition_vars(self, y, x, a):
        return np.array([t.reward + gamma * self.V[t.state.y, t.state.x].var for t in self.transitions(y, x, a)])


def extract_distribution(transitions, var_values, atom_p):
    """

    :param transitions:
    :param var_values:
    :param atom_p:
    :return: sorted list of tuples (probability, index, var)
    """
    info = np.empty(var_values.shape, dtype=object)
    for i_t, t in enumerate(transitions):
        for i_v, v, p_ in zip(range(len(var_values[i_t])), var_values[i_t], atom_p):
            info[i_t, i_v] = (p_ * t.prob, i_t, v)

    info = list(info.flatten())
    info.sort(key=lambda x: x[-1])
    return info



class MarkovState:

    def __init__(self):
        self.nb_atoms = NB_ATOMS
        self.var = np.zeros(self.nb_atoms)
        self.atoms = spaced_atoms(self.nb_atoms)    # e.g. [0, 0.25, 0.5, 1]
        self.atom_p = self.atoms[1:] - self.atoms[:-1]  # [0.25, 0.25, 0.5]

    def plot(self, show=True):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        # ax[1].plot(self.atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')
        if show:
            plt.show()

    def y_cvar(self, alpha, var=None):

        if var is None:
            var = self.var

        if alpha == 0:
            return min(var)

        p = 0.
        cv = 0.
        for p_, v_ in zip(self.atom_p, var):
            if p + p_ >= alpha:
                cv += (alpha-p) * v_
                break
            else:
                cv += p_ * v_
                p += p_

        return cv

    def y_var(self, var, var_values=None):

        if var_values is None:
            var_values = self.var

        cv = 0.
        for p_, v_ in zip(self.atom_p, var_values):
            if v_ > var:  # TODO: solve for discrete distributions
                break
            else:
                cv += p_ * v_

        return cv

    def expected_value(self):
        return np.dot(self.atom_p, self.var)

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

    def compute_wasserstein(self, transition_p, var_values):
        # 0) weight by transition probs
        p = np.outer(transition_p, self.atom_p).flatten()

        # 1) create quantile function
        sortargs = var_values.flatten().argsort()
        var_sorted = var_values.flatten()[sortargs]
        p_sorted = p.flatten()[sortargs]
        p_sorted, var_sorted = further_split(p_sorted, var_sorted, self.atoms)

        # 2) weighted median minimizes wasserstein
        cp = 0.
        var_solution = []
        atom_ix = 0
        for ix, p_, v_ in zip(range(len(p_sorted)), p_sorted, var_sorted):

            median_p = self.atoms[atom_ix] + self.atom_p[atom_ix] / 2

            # if there is a step near the middle, use the midpoint
            if abs(cp + p_ - median_p) < self.atom_p[atom_ix] / 100:
                var_solution.append((v_ + var_sorted[ix + 1]) / 2)
                atom_ix += 1
            # if we get over median, this must be it's value
            elif cp + p_ > median_p:
                atom_ix += 1
                var_solution.append(v_)

            cp += p_

            if atom_ix == self.nb_atoms:
                break
        return var_solution, np.array([self.y_cvar(alpha, var_solution) for alpha in self.atoms[1:]])

    def compute_cvar_by_lp(self, transition_p, var_values):
        """
        Create LP:
        min Sum p_t * I

        0 <= xi <= 1/alpha
        Sum p_t * xi == 1

        I = max{yV}

        return yV[alpha]
        """

        def single_lp(alpha):

            nb_transitions = len(transition_p)

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

            return value(prob.objective)

        yV = [single_lp(alpha) for alpha in self.atoms[1:]]
        # extract vars:
        last = 0.
        new_var = np.zeros_like(yV)
        for i in range(self.nb_atoms):
            ddalpha = (yV[i] - last) / self.atom_p[i]
            new_var[i] = ddalpha
            last = yV[i]

        return new_var, yV


def value_update(world, V):

    V_ = copy.deepcopy(V)
    for s in world.states():
        V_.update(s.y, s.x)

    return V_


def converged(V, V_, world):
    eps = 1e-4
    max_val = eps
    for s in world.states():
        # dist = np.max(np.abs(V.V[s.y, s.x].var-V_.V[s.y, s.x].var))
        cvars = np.array([V.V[s.y, s.x].y_cvar(alpha)*alpha for alpha in V.V[s.y, s.x].atoms])
        cvars_ = np.array([V_.V[s.y, s.x].y_cvar(alpha)*alpha for alpha in V_.V[s.y, s.x].atoms])
        dist = np.max(np.abs(cvars - cvars_))
        max_val = max(max_val, dist)
    if max_val > eps:
        print(max_val)
        # print(s)
        # print(V.V[s.y, s.x].var)
        # print(V_.V[s.y, s.x].var)
        return False
    return True


def value_iteration(world, max_iters=1e3):
    V = ValueFunction(world)
    i = 0
    while True:
        V_ = value_update(world, V)
        # if i % 10 == 0:
        #     V_.V[0, 5].plot()
        if (converged(V, V_, world) and i != 0) or i > max_iters:
            print("value fully learned after %d iterations" % (i,))
            break
        V = V_
        i += 1

        print('Value iteration:', i)

    return V


# TODO: control error by adding atoms
# TODO: smart convergence  ---  the cvar converges, not necessarily the distributions (?)
# TODO: V -> Q

if __name__ == '__main__':

    # world = GridWorld(1, 3, random_action_p=0.3)
    world = GridWorld(4, 6, random_action_p=0.1)

    print('ATOMS:', spaced_atoms(NB_ATOMS))

    # =============== PI setup
    alpha = 0.1
    V = value_iteration(world, max_iters=100)
    # V.V[3,0].plot()
    # print(V.V[1,5].var)
    # print(V.V[3,0].y_cvar(1.0))
    # print(V.V[3,0].expected_value())




