import numpy as np

np.random.seed(1337)
np.set_printoptions(3)

gamma = 0.95


# TODO: what's this?
MIN_VALUE = -60
MAX_VALUE = 0


# TODO: move?
# atom spacing
NB_ATOMS = 15
LOG = True  # atoms are log-spaced
SPACING = 2

# use LP when computing CVaRs
TAMAR_LP = False

WASSERSTEIN = False
# WASSERSTEIN = True
