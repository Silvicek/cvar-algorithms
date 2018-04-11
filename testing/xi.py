import numpy as np
from util import cvar_computation

atoms = np.array([0, 1/4, 1/2, 1])

transition_p = np.array([0.25, 0.75])

yc_values = np.array([[-0.25, -0.25, 0], [-3/4, -5/4, -7/4]])

t_atoms = np.array([atoms, atoms])

alpha = 1
nb_transitions = len(transition_p)
nb_atoms = len(atoms)-1

var_values = np.array([cvar_computation.yc_to_var(atoms, yc) for yc in yc_values])


def check_xis(next_alphas, yc):
    yc_from_xis = 0
    for ix, alp in enumerate(next_alphas):
        t_p = transition_p[ix]
        v_sorted = var_values[ix]
        p_sorted = t_atoms[ix][1:] - t_atoms[ix][:-1]
        yc_from_xis += t_p * cvar_computation.single_alpha_to_yc(p_sorted, v_sorted, alp)

    if abs(yc_from_xis - yc) > 1e-6:
        print("ERROR: xis don't mach up", yc_from_xis, yc)


for alpha in [1/4, 1/2, 3/4]:
    _, yc1, xis1 = cvar_computation.single_var_yc_xis_from_t(transition_p, t_atoms, var_values, alpha)

    yc2, xis2 = cvar_computation.single_yc_lp_from_t(transition_p, t_atoms, yc_values, alpha, xis=True)

    print('alpha={}, yc1={}, yc2={}\nxi1={}\nxi2={}'.format(alpha, yc1, yc2, xis1, xis2))
    check_xis(xis1, yc1)
    # check_xis(xis2, yc2)
    print('---------------------')

