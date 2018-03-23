"""
Different CVaR computations and conversions.
Naming conventions: v=VaR, c=CVaR, yc=yCVaR, t=transition

single_: extract a single value from a distribution


"""
import numpy as np
from pulp import *


# ===================================================================
# Single:
# gets the desired values from a single distribution
# ===================================================================

def single_var_to_alpha(p_sorted, v_sorted, s):
    """ """
    e_min = 0
    ix = 0
    alpha = 0
    for v, p in zip(v_sorted, p_sorted):
        if v > s:
            break
        else:
            ix += 1
            e_min += p*v
            alpha += p

    return alpha


def single_alpha_to_var(p_sorted, v_sorted, alpha):
    p = 0.
    for p_, v_ in zip(p_sorted, v_sorted):
        p += p_
        if p >= alpha:
            return v_
    # numerical 1 != 1
    return v_sorted[-1]


def single_alpha_to_cvar(p_sorted, v_sorted, alpha):
    if alpha == 0:
        return v_sorted[0]

    print('using new implementation')

    p = 0.
    cv = 0.
    for p_, v_ in zip(p_sorted, v_sorted):
        if p + p_ >= alpha:
            cv += (p - alpha)*v_
        else:
            p += p_
            cv += p_*v_

    return cv / alpha


# ===================================================================
# Single from transitions:
# gets the desired values from transition distributions
# ===================================================================

def single_var_yc_xis_from_t(transition_p, t_atoms, var_values, alpha):
    """
    Compute VaR, CVaR and xi values, using uniform last probabilities.

    """

    info = extract_distribution(transition_p, t_atoms, var_values)

    xis = np.zeros(len(transition_p))
    p = 0.
    cv = 0.
    v = 0.

    v_alpha = single_alpha_to_var([p_ for p_, i_t, v in info], [v for p_, i_t, v in info], alpha)
    ix = 0
    for ix, (p_, t_i, v) in enumerate(info):
        if v >= v_alpha:
            cv += (alpha - p) * v
            break
        else:
            xis[t_i] += p_
            cv += p_ * v
            p += p_

    # uniform last atom
    last_v_info = []
    while v == v_alpha and ix < len(info)-1:
        last_v_info.append(info[ix])
        ix += 1
        p_, t_i, v = info[ix]
    last_v_p = np.array([p_ for p_, t_i, v in last_v_info])
    fractions = last_v_p/np.sum(last_v_p)

    for fr, (p_, t_i, v) in zip(fractions, last_v_info):
        xis[t_i] += (alpha - p) * fr

    return v_alpha, cv, xis / transition_p


def single_yc_lp_from_t(transition_p, t_atoms, yc_values, alpha, xis=False):
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

    nb_transitions = len(transition_p)

    Xi = [LpVariable('xi_' + str(i)) for i in range(nb_transitions)]
    I = [LpVariable('I_' + str(i)) for i in range(nb_transitions)]

    prob = LpProblem(name='tamar')

    for xi in Xi:
        prob.addConstraint(0 <= xi)
        prob.addConstraint(xi <= 1./alpha)
    prob.addConstraint(sum([xi*p for xi, p in zip(Xi, transition_p)]) == 1)

    for xi, i, yc, atoms in zip(Xi, I, yc_values, t_atoms):
        last_yc = 0.
        f_params = []
        atom_p = atoms[1:] - atoms[:-1]
        for ix in range(len(yc)):
            # linear interpolation as a solution to 'y = kx + q'
            k = (yc[ix]-last_yc)/atom_p[ix]

            q = last_yc - k * atoms[ix]
            prob.addConstraint(i >= k * xi * alpha + q)
            f_params.append((k, q))
            last_yc = yc[ix]

    # opt criterion
    prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

    prob.solve()

    if xis:
        return value(prob.objective), [value(xi)*alpha for xi in Xi]
    else:
        return value(prob.objective)


# ===================================================================
# Distribution <=> vector
# ===================================================================

def yc_to_var(atoms, y_cvar):
    """ yCVaR -> distribution. Outputs same atoms as input. """
    last = 0.
    var = np.zeros_like(y_cvar)

    for i in range(len(atoms) - 1):
        p = atoms[i + 1] - atoms[i]
        ddy = (y_cvar[i] - last) / p
        var[i] = ddy
        last = y_cvar[i]

    return var


def var_to_ycvar(p_sorted, v_sorted):  # TODO: name
    yc = np.zeros_like(p_sorted)
    yc_last = 0
    for i in range(len(yc)):
        yc[i] = yc_last + p_sorted[i] * v_sorted[i]
        yc_last = yc[i]
    return yc


def var_vector(atoms, p_sorted, v_sorted):
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


def ycvar_vector(atoms, p_sorted, v_sorted):
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


# ===================================================================
# Transitions => vector
# ===================================================================


def v_yc_from_t_lp(atoms, transition_p, yc_values):
    """ CVaR computation by dual decomposition LP. """
    y_cvar = [single_yc_lp_from_t(atoms, transition_p, yc_values, alpha) for alpha in atoms[1:]]
    # extract vars:
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def v_yc_from_t(atoms, transition_p, var_values, t_atoms):
    """
    CVaR computation by using underlying distributions.
    :param atoms: points of interest
    :param transition_p:
    :param var_values: (transitions, nb_atoms)
    :param t_atoms: (transitions, nb_atoms+1) e.g. [0, 0.25, 0.5, 1]
    :return:
    """
    # 0) weight by transition probs
    p = np.concatenate([transition_p[i]*(t_atoms[i][1:] - t_atoms[i][:-1]) for i in range(len(transition_p))])

    # 1) sort
    sortargs = np.concatenate(var_values).argsort()
    var_sorted = np.concatenate(var_values)[sortargs]
    p_sorted = p[sortargs]

    # 2) compute y_cvar for each atom
    y_cvar = ycvar_vector(atoms, p_sorted, var_sorted)

    # 3) get vars from y_cvar
    var = yc_to_var(atoms, y_cvar)

    return var, y_cvar


def v_0_from_transitions(V, transitions, gamma):
    return min([t.reward + gamma*V[t.state.y, t.state.x].c_0 for t in transitions])


def extract_distribution(transition_p, t_atoms, var_values):
    """
    :return: sorted list of tuples (probability, index, var)
    """
    info = []
    for i_t, t_p in enumerate(transition_p):
        for v, p_ in zip(var_values[i_t], t_atoms[i_t][1:]-t_atoms[i_t][:-1]):
            info.append((p_ * t_p, i_t, v))

    info.sort(key=lambda x: x[-1])
    return info


# ===================================================================
# Other
# ===================================================================

def var_cvar_from_samples(samples, alpha):
    samples = np.sort(samples)
    alpha_ix = int(np.round(alpha * len(samples)))
    var = samples[alpha_ix - 1]
    cvar = np.mean(samples[:alpha_ix])
    return var, cvar
