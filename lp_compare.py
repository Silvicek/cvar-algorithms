""" Separate file used for early tests and plots. """
# TODO: delete/merge this file?
import matplotlib.pyplot as plt
from cycler import cycler
from pulp import *

from util.cvar_computation import s_to_alpha

plt.rc('axes', prop_cycle=(cycler('color', ['#1f77b4', '#d62728', '#009900'])))

# tex
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# ====================


def tamar_lp_single(alpha):
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

    Xi = [LpVariable('xi_' + str(i)) for i in range(nb_transitions)]
    I = [LpVariable('I_' + str(i)) for i in range(nb_transitions)]

    prob = LpProblem(name='tamar')

    for xi in Xi:
        prob.addConstraint(0 <= xi)
        prob.addConstraint(xi <= 1./alpha)
    prob.addConstraint(sum([xi*p for xi, p in zip(Xi, transition_p)]) == 1)

    for xi, i, var in zip(Xi, I, var_values):
        last = 0.
        f_params = []
        for ix in range(nb_atoms):
            k = var[ix]
            last += k * atom_p[ix]
            q = last - k * atoms[ix+1]
            prob.addConstraint(i >= k * xi * alpha + q)
            f_params.append((k,q))

    # opt criterion
    prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

    prob.solve()

    return value(prob.objective)


def tamar_lp():
    yV = [tamar_lp_single(alpha) for alpha in atoms[1:]]
    # extract vars:
    last = 0.
    var_solution = np.zeros_like(yV)
    for i in range(nb_atoms):
        ddalpha = (yV[i] - last) / atom_p[i]
        var_solution[i] = ddalpha
        last = yV[i]

    return var_solution


def wasserstein_lp():
    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) create quantile function
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]
    p_sorted, var_sorted = further_split(p_sorted, var_sorted)


    # 2) create LP minimizing |y-var|
    Y = [LpVariable('y_'+str(i)) for i in range(nb_atoms)]
    U = [LpVariable('u_'+str(i)) for i in range(len(p_sorted))]  # abs value slack

    prob = LpProblem(name='wasserstein')

    cp = 0.
    atom_ix = 1
    for u, p_, v_ in zip(U, p_sorted, var_sorted):
        cp += p_

        prob.addConstraint(u >= Y[atom_ix-1] - v_)
        prob.addConstraint(u >= v_ - Y[atom_ix-1])

        if cp == atoms[atom_ix]:
            atom_ix += 1

    # opt criterion
    prob.setObjective(sum([u*p for u, p in zip(U, p_sorted)]))

    prob.solve()

    print(value(prob.objective))

    return [value(y_) for y_ in Y]


def wasserstein_median():
    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) create quantile function
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]
    p_sorted, var_sorted = further_split(p_sorted, var_sorted)

    # 2) median minimizes wasserstein
    cp = 0.
    var_solution = []
    atom_ix = 0
    for ix, p_, v_ in zip(range(len(p_sorted)), p_sorted, var_sorted):

        median_p = atoms[atom_ix] + atom_p[atom_ix]/2

        if abs(cp + p_ - median_p) < atom_p[atom_ix]/100:  # there is a step near the middle
            var_solution.append((v_ + var_sorted[ix+1])/2)
            atom_ix += 1
        elif cp + p_ > atoms[atom_ix] + atom_p[atom_ix]/2:
            atom_ix += 1
            var_solution.append(v_)

        cp += p_

        if atom_ix == nb_atoms:
            break

    return var_solution

#
# def further_split(p, v):
#     cp = 0.
#     atom_ix = 1
#     new_p = []
#     new_v = []
#
#     for ix, (p_, v_) in enumerate(zip(p, v)):
#         while abs(p_) > 1e-5:
#             if cp+p_ >= atoms[atom_ix]:
#                 p__ = atoms[atom_ix] - cp
#                 p_ = p_ - p__
#                 atom_ix += 1
#                 cp += p__
#
#                 new_p.append(p__)
#                 new_v.append(v_)
#
#             else:
#                 cp += p_
#                 new_p.append(p_)
#                 new_v.append(v_)
#                 p_ = 0
#
#     # print('------------------')
#     # print(new_p)
#     # print(new_v)
#     return new_p, new_v


def exact_pv():
    p = np.outer(transition_p, atom_p).flatten()

    # 1) sort
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]
    return p_sorted, var_sorted


def simple_sort():
    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) sort
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]

    p_sorted, var_sorted = further_split(p_sorted, var_sorted)

    # 2) compute yV for each atom

    yV = np.zeros(nb_atoms)
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
    for i in range(nb_atoms):
        ddalpha = (yV[i]-last)/atom_p[i]
        var_solution[i] = ddalpha
        last = yV[i]

    return var_solution


def plot(*solutions, legend=True):
    # solution = (name, (prob, var))

    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    axs = np.array(axs)
    axs = axs.reshape(-1)

    # var
    ax = axs[0]
    for _, (p, sol) in solutions:
        sol = list(sol)
        print(atoms)
        print(sol + [sol[-1]])
        ax.step(np.insert(np.cumsum(p), 0, 0), sol + [sol[-1]], where='post')
    ax.set_title('Quantile function')

    # yV
    ax = axs[1]
    for _, (p, sol) in solutions:
        ax.plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * sol), 0, 0), 'o-')
    ax.set_title('$\\alpha$CVaR')

    # cvar
    ax = axs[2]
    for _, (p, sol) in solutions:
        p, v = var_to_cvar_approx(p, sol)
        ax.plot(p, v)
    ax.set_title('CVaR')

    # cvar_s
    ax = axs[3]
    for _, (p, sol) in solutions:
        a = [s_to_alpha(s, p, sol) for s in s_range]
        v = []
        ax.plot(s_range, v)
    a = [s_to_alpha(s, atom_p, wm) for s in s_range]
    cv = []

    ax.set_title('CVaR(s)')

    # =====================================================

    # legend
    if legend:
        for ax in axs:
            ax.legend([name for name, _ in solutions])

    # hide last plot
    # ax[1][1].axis('off')

    # grid: on
    for ax in axs:
        ax.grid()

    # hide upper x axis
    # plt.setp(ax[0].get_xticklabels(), visible=False)

    plt.show()
    # plt.savefig('files/exactvarcvar.pdf')


def var_to_cvar_approx(p, var, res=0.001):

    cvar = np.zeros(int(1/res))

    cp = 0.
    ccp = 0.
    cv = 0.
    ix = 0
    for p_, v_ in zip(p, var):

        while ccp < min(1, cp+p_):

            ccp += res
            cv += res*v_
            cvar[ix] = cv / ccp
            ix += 1
        cp = ccp
    return np.arange(res, 1+res, res), cvar


def plot_process():
    plt.rcParams['axes.grid'] = True

    fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharey=True)


    # var
    p, v = atom_p, var_values[0]
    ax[0][0].step(np.insert(np.cumsum(p), 0, 0), np.insert(v, 0, v[0]), 'o-', where='pre')
    p, v = atom_p, var_values[1]
    ax[0][1].step(np.insert(np.cumsum(p), 0, 0), np.insert(v, 0, v[0]), 'o-', where='pre')
    p, v = exact_pv()
    ax[0][2].step(np.insert(np.cumsum(p), 0, 0), np.insert(v, 0, v[0]), 'o-', where='pre')
    p, v = atom_p, tamar_lp()
    ax[0][3].step(np.insert(np.cumsum(p), 0, 0), np.insert(v, 0, v[0]), 'o-', where='pre')

    plt.savefig('files/multivar.pdf')
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    # yCVaR
    p, v = atom_p, var_values[0]
    ax[1][0].plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * v), 0, 0), 'o-')
    p, v = atom_p, var_values[1]
    ax[1][1].plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * v), 0, 0), 'o-')
    p, v = exact_pv()
    ax[1][2].plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * v), 0, 0), 'o-')
    p, v = atom_p, tamar_lp()
    ax[1][3].plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * v), 0, 0), 'o-')

    plt.savefig('files/multiycvar.pdf')
    plt.show()


if __name__ == '__main__':
    # nb_atoms = 3
    # nb_transitions = 2
    #
    # transition_p = np.array([0.25, 0.75])
    #
    # atoms = np.array([0., 0.25, 0.5, 1.])
    # atom_p = atoms[1:] - atoms[:-1]
    #
    # var_values = np.array([[-1, 0, 0.5],
    #                        [-3, -2, -1]])

    nb_atoms = 4
    nb_transitions = 2

    transition_p = np.array([0.25, 0.75])

    atoms = np.array([0., 0.25, 0.5, 0.75, 1.])
    atom_p = atoms[1:] - atoms[:-1]

    var_values = np.array([[-0.5, 0.25, 0.5, 1],
                           [-3, -2, -1, 0]])

    # ================================================

    # nb_atoms = 4
    # nb_transitions = 2
    # var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
    # var_values.sort()
    #
    # transition_p = softmax(np.random.random(nb_transitions))
    # atoms = np.zeros(nb_atoms + 1)
    # atoms[1:] = np.cumsum(softmax(np.random.random(nb_atoms)))
    # atom_p = atoms[1:] - atoms[:-1]
    #
    # var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
    # var_values.sort()

    print(atoms)
    print(atom_p)
    print(var_values)
    print('-----------------------')

    ss = simple_sort()
    wm = wasserstein_median()
    tam = tamar_lp()

    ex_p, ex_v = exact_pv()

    print('sort:', ss)
    print('wasserstein med:', wm)
    print('tamar:', tam)

    s_range = np.arange(ex_v[0], ex_v[-1], 0.01)
    # plt.plot(s_range, [cvar_s(s, ss, atom_p) for s in s_range])
    # plt.show()
    # quit()


    # plot(exact_pv(), ('sort', ss), ('wasserstein', wm), ('tamar', tam))
    plot(('Exact', (ex_p, ex_v)), ('CVaR VI', (atom_p, tam)), ('Wasserstein', (atom_p, wm)))
    # plot(('Exact', (ex_p, ex_v)), ('CVaR VI', (atoms, tam)), ('Wasserstein', (atoms, wm)))
    # plot_process()







