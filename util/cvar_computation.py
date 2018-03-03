from util import *


def cvar_from_samples(samples, alpha):
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
        # return var_values[0]
    else:
        return alpha
        # return e_min + v*alpha