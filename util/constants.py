import numpy as np
from util.util import spaced_atoms

np.random.seed(9)
np.set_printoptions(3)
print("seed")

gamma = 0.95


# atom spacing
NB_ATOMS = 10
LOG = True  # atoms are log-spaced
SPACING = 2

atoms = spaced_atoms(NB_ATOMS, SPACING, LOG)    # e.g. [0, 0.25, 0.5, 1]
atom_p = atoms[1:] - atoms[:-1]  # [0.25, 0.25, 0.5]

