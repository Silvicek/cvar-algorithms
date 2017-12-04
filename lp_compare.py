import numpy as np
from pulp import *
import matplotlib.pyplot as plt
# np.random.seed(4)

def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)

nb_atoms = 3
nb_transitions = 2

transition_p = np.array([0.25, 0.75])

atoms = np.array([0., 0.25, 0.5, 1.])
atom_p = atoms[1:] - atoms[:-1]

var_values = np.array([[-1, 0, 0.5],
                      [-3, -2, -1]])



nb_atoms = 4
nb_transitions = 2
var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
var_values.sort()

transition_p = softmax(np.random.random(nb_transitions))
atoms = np.zeros(nb_atoms+1)
atoms[1:] = np.cumsum(softmax(np.random.random(nb_atoms)))
atom_p = atoms[1:] - atoms[:-1]

var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
var_values.sort()

print(atoms)
print(atom_p)
print(var_values)


def tamar_lp():
    pass


def wasserstein():
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
    for p_, v_ in zip(p_sorted, var_sorted):
        cp += p_
        if cp >= atoms[atom_ix] + atom_p[atom_ix]/2:
            atom_ix += 1
            var_solution.append(v_)

        if atom_ix == nb_atoms:
            break

    return var_solution


def further_split(p, v):
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

    # print('------------------')
    # print(new_p)
    # print(new_v)
    return new_p, new_v


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

    # print('--------------')
    # print(yV)
    # print(-11/16, -17/16, -21/16)
    # 3) get vars from yV
    last = 0.
    var_solution = np.zeros_like(yV)
    for i in range(nb_atoms):
        ddalpha = (yV[i]-last)/atom_p[i]
        var_solution[i] = ddalpha
        last = yV[i]

    return var_solution


def plot(exact, *solutions):

    fig, ax = plt.subplots(2, 2)

    p, v = exact

    # var
    ax[0][0].step(np.insert(np.cumsum(p), 0, 0), np.insert(v, 0, v[0]), where='pre')
    for sol in solutions:
        sol = list(sol)
        ax[0][0].step(atoms, sol + [sol[-1]], where='post')

    ax[0][0].legend(['exact', 'simple_sort', 'wasserstein LP', 'wasserstein median'])


    # yV
    ax[0][1].plot(np.insert(np.cumsum(p), 0, 0), np.insert(np.cumsum(p * v), 0, 0), 'o-')
    for sol in solutions:
        ax[0][1].plot(atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')

    ax[0][1].legend(['exact', 'simple_sort', 'wasserstein LP', 'wasserstein median'])

    # cvar
    p, v = var_to_cvar(p, v)
    ax[1][0].plot(p, v)
    for sol in solutions:
        p, v = var_to_cvar(atom_p, sol)
        ax[1][0].plot(p, v)

    ax[1][0].legend(['exact', 'simple_sort', 'wasserstein LP', 'wasserstein median'])

    plt.show()




def var_to_cvar(p, var, res=0.01):

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


ss = simple_sort()
w = wasserstein()
wm = wasserstein_median()

print('sort:', ss)
print('wasserstein LP:', w)
print('wasserstein med:', wm)



plot(exact_pv(), ss, w, wm)
