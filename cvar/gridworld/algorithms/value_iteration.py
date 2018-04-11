import copy
from cvar.gridworld.cliffwalker import *
from cvar.gridworld.core import cvar_computation
from cvar.gridworld.core.constants import *
from cvar.common.util import timed, spaced_atoms

# use LP when computing CVaRs
# TAMAR_LP = True
TAMAR_LP = False

class ValueFunction:

    def __init__(self, world):
        self.world = world

        self.V = np.empty((world.height, world.width), dtype=object)
        for ix in np.ndindex(self.V.shape):
            self.V[ix] = MarkovState()

        print('ATOMS:', list(self.V[0, 0].atoms))

    def update(self, y, x, check_bound=False):

        v_a, yc_a = self.action_v_yc(y, x)

        best_args = np.argmax(yc_a, axis=0)

        self.V[y, x].yc = np.array([yc_a[best_args[i], i] for i in range(len(self.V[y, x].yc))])

        self.V[y, x].c_0 = max([cvar_computation.v_0_from_transitions(self.V, list(self.transitions(y, x, a)), gamma)
                                for a in self.world.ACTIONS])

        # check for error bound
        if check_bound:
            eps = 1.
            c_0 = v_a[best_args[0], 0]
            if c_0 - self.V[y, x].c_0 > eps:
                # if deep and self.V[y, x].nb_atoms < 100:
                self.V[y, x].increase_precision(eps)

    def action_v_yc(self, y, x):
        """ Extract transition distributions for each action. """
        yc_a = []
        v_a = []

        for a in self.world.ACTIONS:
            t = list(self.transitions(y, x, a))

            if TAMAR_LP:
                v, yc = self.V[y, x].compute_cvar_by_lp([t_.prob for t_ in t], self.transition_ycs(y, x, a),
                                                        [self.V[tr.state.y, tr.state.x].atoms for tr in t])
            else:
                v, yc = self.V[y, x].compute_cvar_by_sort([t_.prob for t_ in t], self.transition_vars(y, x, a),
                                                          [self.V[tr.state.y, tr.state.x].atoms for tr in t])
            yc_a.append(yc)
            v_a.append(v)

        return np.array(v_a), np.array(yc_a)

    def next_action(self, y, x, alpha):
        if alpha == 0:
            print('alpha=0')
            a_best = max(self.world.ACTIONS,
                         key=lambda a:cvar_computation.v_0_from_transitions(self.V, list(self.transitions(y, x, a)), gamma))
            return a_best, np.zeros(len(list(self.transitions(y, x, a_best))))

        assert alpha != 0
        # self.plot_full_actions(31, 59)
        best = (-1e6, 0, 0)
        for a in self.world.ACTIONS:

            if TAMAR_LP:
                cv, xis = self.single_yc_xis_lp(y, x, a, alpha)
            else:
                _, cv, xis = self.single_var_yc_xis(y, x, a, alpha)

            if cv > best[0]:
                best = (cv, xis, a)

        _, xis, a = best
        return a, xis

    def single_yc_xis_lp(self, y, x, a, alpha):
        transition_p = [t.prob for t in self.transitions(y, x, a)]
        atom_values = [self.V[t.state.y, t.state.x].atoms for t in self.transitions(y, x, a)]
        yc = self.transition_ycs(y, x, a)
        return cvar_computation.single_yc_lp_from_t(transition_p, atom_values, yc, alpha, xis=True)

    def single_var_yc_xis(self, y, x, a, alpha):
        """
        Compute VaR, CVaR and xi values in O(nlogn)
        """

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)
        transition_p = [t.prob for t in transitions]
        t_atoms = [self.V[t.state.y, t.state.x].atoms for t in transitions]

        return cvar_computation.single_var_yc_xis_from_t(transition_p, t_atoms, var_values, alpha)

    def plot_full_actions(self, y, x):
        """
        Plot actions without value approximation - used for debugging
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for a in self.world.ACTIONS:
            transitions = list(self.transitions(y, x, a))
            var_values = self.transition_vars(y, x, a)
            transition_p = [t.prob for t in transitions]
            t_atoms = [self.V[t.state.y, t.state.x].atoms for t in transitions]

            info = cvar_computation.extract_distribution(transition_p, t_atoms, var_values)
            t_atoms = np.cumsum([0]+[p for p, ix, v in info])
            t_vars = [v for p, ix, v in info]
            t_yc = cvar_computation.var_to_ycvar([p for p, ix, v in info], t_vars)
            print(a, cvar_computation.single_alpha_to_yc([p for p, ix, v in info], t_vars, 0.036))

            ax.plot(t_atoms, [0]+list(t_yc), 'o-')

        ax.legend([self.world.ACTION_NAMES[a] for a in self.world.ACTIONS])

        plt.show()

    def y_var(self, y, x, a, var):
        """ E[(Z-var)^-] + yvar"""

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)

        info = cvar_computation.extract_distribution(transitions, var_values,
                                                     [self.V[tr.state.y, tr.state.x].atom_p for tr in transitions])

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

    def transition_ycs(self, y, x, a):
        return np.array([t.reward*self.V[t.state.y, t.state.x].atoms[1:] + gamma * self.V[t.state.y, t.state.x].yc for t in self.transitions(y, x, a)])

    def optimal_path(self, alpha):
        """ Optimal deterministic path. """
        from cvar.gridworld.core.policies import XiBasedPolicy
        from cvar.gridworld.core.runs import optimal_path
        policy = XiBasedPolicy(self, alpha)
        return optimal_path(self.world, policy)

    def plot(self, y, x, a, show=False, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 2)

        var_a, yc_a = self.action_v_yc(y, x)
        var = var_a[a]
        yc = yc_a[a]
        # var
        ax[0].step(self.V[y, x].atoms, list(var) + [var[-1]], 'o-', where='post')

        # yV
        ax[1].plot(self.V[y, x].atoms, np.insert(yc, 0, 0), 'o-')
        if show:
            plt.show()


class MarkovState:

    def __init__(self):
        self.yc = np.zeros(NB_ATOMS)
        self.atoms = spaced_atoms(NB_ATOMS, SPACING, LOG_NB_ATOMS, LOG_THRESHOLD)    # e.g. [0, 0.25, 0.5, 1]
        self.atom_p = self.atoms[1:] - self.atoms[:-1]  # [0.25, 0.25, 0.5]

        self.c_0 = 0  # separate estimate for CVaR_0

    def plot(self, show=True, figax=None):
        import matplotlib.pyplot as plt
        if figax is None:
            fig, ax = plt.subplots(1, 2)
        else:
            fig, ax = figax

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        ax[1].plot(self.atoms, np.insert(self.yc, 0, 0), 'o-')
        if show:
            plt.show()

    @property
    def var(self):
        return cvar_computation.yc_to_var(self.atoms, self.yc)

    @property
    def nb_atoms(self):
        return len(self.yc)

    def increase_precision(self, eps):
        """ Bound error by adding atoms. Follows the adaptive procedure from RSRDM. """
        new_atoms = []
        v_0 = self.yc[0]
        y = (eps*self.atom_p[0])/(np.abs(v_0-self.c_0))
        if y < 1e-15:
            print('SMALL')

        while y < self.atom_p[0]:
            new_atoms.append(y)
            y *= SPACING

        self.atoms = np.hstack((np.array([0]), new_atoms, self.atoms[1:]))
        self.atom_p = self.atoms[1:] - self.atoms[:-1]

        self.yc = np.hstack((v_0*np.array(new_atoms), self.yc))

    def cvar_alpha(self, alpha):
        return cvar_computation.single_alpha_to_cvar(self.atom_p, self.var, alpha)

    def expected_value(self):
        return np.dot(self.atom_p, self.var)

    def compute_cvar_by_sort(self, transition_p, var_values, t_atoms):
        return cvar_computation.v_yc_from_t(self.atoms, transition_p, var_values, t_atoms)

    def compute_cvar_by_lp(self, transition_p, t_ycs, t_atoms):
        return cvar_computation.v_yc_from_t_lp(self.atoms, transition_p, t_ycs, t_atoms)


def value_update(world, V, id=0, figax=None):

    V_ = copy.deepcopy(V)
    for s in world.states():
        V_.update(s.y, s.x)

    return V_


def value_difference(V, V_, world):
    max_val = -1
    max_state = None
    for s in world.states():
        # dist = np.max(np.abs(V.V[s.y, s.x].var-V_.V[s.y, s.x].var))
        cvars = np.array([V.V[s.y, s.x].cvar_alpha(alpha) for alpha in V.V[s.y, s.x].atoms[1:]])
        cvars_ = np.array([V_.V[s.y, s.x].cvar_alpha(alpha) for alpha in V_.V[s.y, s.x].atoms[1:]])
        if cvars.shape != cvars_.shape:
            return float('inf')
        dist = np.max(np.abs(cvars - cvars_))
        if dist > max_val:
            max_state = s
            max_val = dist

    return max_val, max_state

@timed
def value_iteration(world, V=None, max_iters=1e3, eps_convergence=1e-3):
    if V is None:
        V = ValueFunction(world)
    i = 0
    figax = None
    while True:
        if i == 28:
            import matplotlib.pyplot as plt
            figax = plt.subplots(1, 2)
        V_ = value_update(world, V, i, figax)

        error, worst_state = value_difference(V, V_, world)
        if error < eps_convergence:
            print("value fully learned after %d iterations" % (i,))
            break
        elif i > max_iters:
            print("value finished without convergence after %d iterations" % (i,))
            break
        V = V_
        i += 1

        print('Iteration:{}, error={} ({})'.format(i, error, worst_state))

    return V


if __name__ == '__main__':
    import pickle
    from cvar.gridworld.plots.grid import InteractivePlotMachine
    from cvar.common.util import tick, tock
    np.random.seed(2)
    # ============================= new config
    tick()
    world = GridWorld(40, 60, random_action_p=0.05)
    V = value_iteration(world, max_iters=1000)
    pickle.dump((world, V), open('../data/models/vi_test.pkl', mode='wb'))
    tock()
    # ============================= load
    world, V = pickle.load(open('../data/models/vi_test.pkl', 'rb'))

    # ============================= RUN
    for alpha in np.arange(0.05, 1.01, 0.05):
        print(alpha)
        pm = InteractivePlotMachine(world, V, alpha=alpha)
        pm.show()

