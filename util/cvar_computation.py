import numpy as np
from pulp import *

# TODO: unify atoms, atom_p


def v_c_from_samples(samples, alpha):
    samples = np.sort(samples)
    alpha_ix = int(np.round(alpha * len(samples)))
    var = samples[alpha_ix - 1]
    cvar = np.mean(samples[:alpha_ix])
    return var, cvar


def s_to_alpha(s, p_atoms, var_values):
    e_min = 0
    ix = 0
    alpha = 0
    for v, p in zip(var_values, p_atoms):
        if v >= s:
            break
        else:
            ix += 1
            e_min += p*v
            alpha += p

    if ix == 0:
        return 0
    else:
        return alpha


def yc_vector(atoms, p_sorted, v_sorted):
    """
    Compute yCvAR at desired atom locations.
    len(p) == len(var)
    :param atoms: desired atom locations.
    """
    y_cvar = np.zeros_like(atoms)
    p = 0
    ycv = 0
    ix = 0  # index in p,v
    atom_ix = 1
    p_ = 0
    while p < 1 and ix < len(p_sorted):
        if p_ == 0:
            p_ = p_sorted[ix]
        v_ = v_sorted[ix]

        if p + p_ >= atoms[atom_ix]:
            p_difference = atoms[atom_ix] - p
            ycv += p_difference * v_
            p = atoms[atom_ix]
            y_cvar[atom_ix] = ycv
            atom_ix += 1
            p_ -= p_difference
        else:
            ycv += p_ * v_
            p += p_
            ix += 1
            p_ = 0
    assert abs(p-1) < 1e-5

    return y_cvar


def yc_to_var(atoms, y_cvar):
    """ yCVaR -> distribution """
    last = 0.
    var = np.zeros_like(y_cvar)

    for i in range(len(atoms)-1):
        p = atoms[i+1] - atoms[i]
        ddy = (y_cvar[i] - last) / p
        var[i] = ddy
        last = y_cvar[i]

    return var


def var_to_yc(atoms, var):
    pass


def tamar_lp_single(atoms, transition_p, var_values, alpha):
    """
    Create LP:
    min Sum p_t * I

    0 <= xi <= 1/alpha
    Sum p_t * xi == 1

    I = max{y_cvar}

    return y_cvar[alpha]
    """
    if alpha == 0:
        return 0.

    atom_p = atoms[1:] - atoms[:-1]
    nb_atoms = len(atom_p)
    nb_transitions = len(transition_p)

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


def v_yc_from_transitions_lp(atoms, transition_p, var_values):
    # TODO: single LP
    y_cvar = [tamar_lp_single(atoms, transition_p, var_values, alpha) for alpha in atoms[1:]]
    # extract vars:
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def v_yc_from_transitions_sort(atoms, transition_p, var_values):
    atom_p = atoms[1:] - atoms[:-1]
    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) sort
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]

    # 2) compute y_cvar for each atom
    y_cvar = yc_vector(atoms, p_sorted, var_sorted)

    # 3) get vars from y_cvar
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


if __name__ == '__main__':

    # print(yc_vector([0.25, 0.5, 1.], [0.75, 0.25], [1., 2.]))
    print(yc_to_var([0.25, 0.5, 1.], [-0.25, -0.25, 0]))
