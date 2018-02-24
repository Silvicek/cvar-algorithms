import numpy as np

np.random.seed(1337)
np.set_printoptions(3)

MIN_VALUE = -60
MAX_VALUE = 0

gamma = 0.95

# atom spacing
NB_ATOMS = 15
LOG = True
SPACING = 2

# use LP when computing CVaRs
TAMAR_LP = False

WASSERSTEIN = False
# WASSERSTEIN = True


def spaced_atoms(nb_atoms):
    if LOG:
        if SPACING < 2:
            return np.array([0, 0.5 / SPACING ** (nb_atoms-2)] + [1. / SPACING ** (nb_atoms - 1 - i) for i in range(1, nb_atoms)])
        return np.array([0] + [1. / SPACING ** (nb_atoms - 1 - i) for i in range(nb_atoms)])
    else:
        return np.linspace(0, 1, nb_atoms+1)


def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)