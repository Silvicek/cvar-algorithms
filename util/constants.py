import numpy as np
from util.util import spaced_atoms

np.random.seed(5)  # 10, 15
# np.random.seed(6)  # 40, 60
np.set_printoptions(3)
print("seed")

gamma = 0.95


# atom spacing
NB_ATOMS = 21
LOG = True  # atoms are log-spaced
SPACING = 2.065

atoms = spaced_atoms(NB_ATOMS, SPACING, LOG)    # e.g. [0, 0.25, 0.5, 1]
atom_p = atoms[1:] - atoms[:-1]  # [0.25, 0.25, 0.5]

