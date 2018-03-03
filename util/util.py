import numpy as np

# def clip(ix):
#     new_ix = max(0, min(MAX_VALUE - MIN_VALUE, ix))
#     return new_ix


def spaced_atoms(nb_atoms, spacing, log=False):
    if log:
        if spacing < 2:
            return np.array([0, 0.5 / spacing ** (nb_atoms-2)] + [1. / spacing ** (nb_atoms - 1 - i) for i in range(1, nb_atoms)])
        return np.array([0] + [1. / spacing ** (nb_atoms - 1 - i) for i in range(nb_atoms)])
    else:
        return np.linspace(0, 1, nb_atoms+1)


def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)

