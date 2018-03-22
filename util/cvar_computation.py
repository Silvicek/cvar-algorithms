"""
Different CVaR computations and conversions.
Naming conventions: v=VaR, c=CVaR, yc=yCVaR
"""
import numpy as np
from pulp import *


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
        if v > s:
            break
        else:
            ix += 1
            e_min += p*v
            alpha += p

    if ix == 0:
        return 0
    else:
        return alpha


def single_var(p_sorted, v_sorted, alpha):
    if alpha > 0.999:
        return v_sorted[-1]
    p = 0.
    i = 0
    while p < alpha:
        p += p_sorted[i]
        i += 1

    if p == alpha:
        var = v_sorted[i - 1]
    else:
        var = v_sorted[i]

    return var


def v_vector(atoms, p_sorted, v_sorted):
    """
    :param atoms: full atoms, shape=[n]
    :param p_sorted: probability mass, shape=[m]
    :param v_sorted: quantile values, shape=[m]
    :return: VaR at atoms[1:], shape=[n-1]
    """
    v = np.zeros(len(atoms)-1)
    p = 0
    ix = 0  # index in p,v
    atom_ix = 1
    p_ = 0
    while p < 1 and ix < len(p_sorted):
        if p_ == 0:
            p_ = p_sorted[ix]

        if p + p_ >= atoms[atom_ix]:
            p_difference = atoms[atom_ix] - p
            p = atoms[atom_ix]
            v[atom_ix-1] = v_sorted[ix]
            atom_ix += 1
            p_ -= p_difference
        else:
            p += p_
            ix += 1
            p_ = 0
    assert abs(p - 1) < 1e-5

    return v


def single_cvar(p_sorted, v_sorted, alpha):
    # TODO: check
    if alpha == 0:
        return v_sorted[0]
    if alpha > 0.99999:
        return np.dot(p_sorted, v_sorted)
    p = 0.
    i = 0
    while p < alpha:
        p += p_sorted[i]
        i += 1
    i -= 1
    p = p - p_sorted[i]
    p_rest = alpha - p
    cvar = (np.dot(p_sorted[:i], v_sorted[:i]) + p_rest * v_sorted[i]) / alpha

    return cvar


def yc_vector(atoms, p_sorted, v_sorted):
    """
    Compute yCvAR at desired atom locations.
    len(p) == len(var)
    :param atoms: desired atom locations.
    """
    y_cvar = np.zeros(len(atoms)-1)
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
            y_cvar[atom_ix-1] = ycv
            atom_ix += 1
            p_ -= p_difference
            if p_ == 0:
                ix += 1
        else:
            ycv += p_ * v_
            p += p_
            ix += 1
            p_ = 0

    # numerical errors
    if p != 1:
        y_cvar[-1] = ycv

    assert abs(p-1) < 1e-5
    assert abs(y_cvar[-1] - np.dot(p_sorted, v_sorted)) < 1e-5

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


def var_to_yc(p_sorted, v_sorted):
    yc = np.zeros_like(p_sorted)
    yc_last = 0
    for i in range(len(yc)):
        yc[i] = yc_last + p_sorted[i]*v_sorted[i]
        yc_last = yc[i]
    return yc


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
            f_params.append((k, q))

    # opt criterion
    prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

    prob.solve()

    return value(prob.objective)


def v_yc_from_transitions_lp_yc(atoms, transition_p, yc_values):
    """ CVaR computation by dual decomposition LP. """
    y_cvar = [tamar_lp_single_yc(atoms, transition_p, yc_values, alpha) for alpha in atoms[1:]]
    # extract vars:
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def tamar_lp_single_yc(atoms, transition_p, yc_values, alpha):
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

    for xi, i, yc in zip(Xi, I, yc_values):
        last_yc = 0.
        f_params = []
        for ix in range(nb_atoms):
            # linear interpolation as a solution to 'y = kx + q'
            k = (yc[ix]-last_yc)/atom_p[ix]

            q = last_yc - k * atoms[ix]
            prob.addConstraint(i >= k * xi * alpha + q)
            f_params.append((k, q))
            last_yc = yc[ix]

    # opt criterion
    prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

    prob.solve()

    return value(prob.objective)


def v_yc_from_transitions_lp(atoms, transition_p, var_values):
    """ CVaR computation by dual decomposition LP. """
    # TODO: single LP
    y_cvar = [tamar_lp_single(atoms, transition_p, var_values, alpha) for alpha in atoms[1:]]
    # extract vars:
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def v_yc_from_transitions_sort(atoms, transition_p, var_values, t_atoms):
    """
    CVaR computation by using underlying distributions.
    :param transition_p:
    :param var_values: (transitions, nb_atoms)
    :param atoms: (transitions, nb_atoms+1) e.g. [0, 0.25, 0.5, 1]
    :return:
    """
    # 0) weight by transition probs
    p = np.concatenate([transition_p[i]*(t_atoms[i][1:] - t_atoms[i][:-1]) for i in range(len(transition_p))])

    # 1) sort
    sortargs = np.concatenate(var_values).argsort()
    var_sorted = np.concatenate(var_values)[sortargs]
    p_sorted = p[sortargs]

    # 2) compute y_cvar for each atom
    y_cvar = yc_vector(atoms, p_sorted, var_sorted)

    # 3) get vars from y_cvar
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def v_0_from_transitions(V, transitions, gamma):
    return min([t.reward + gamma*V[t.state.y, t.state.x].c_0 for t in transitions])


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


if __name__ == '__main__':

    # print(yc_vector([0.25, 0.5, 1.], [0.75, 0.25], [1., 2.]))
    print(s_to_alpha(-2, [1/3, 1/3, 1/3], [-2, -0, 1]))
    print(s_to_alpha(-1, [1/3, 1/3, 1/3], [-2, -0, 1]))
    print(s_to_alpha(0, [1/3, 1/3, 1/3], [-2, -0, 1]))
    print(s_to_alpha(1, [1/3, 1/3, 1/3], [-2, -0, 1]))
    print(s_to_alpha(2, [1/3, 1/3, 1/3], [-2, -0, 1]))
