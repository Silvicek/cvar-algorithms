import numpy as np
from cvar.gridworld.core.util import spaced_atoms

np.random.seed(2)  # 10, 15
# np.random.seed(6)  # 40, 60
np.set_printoptions(8)
print("seed")

gamma = 0.95


# atom spacing
NB_ATOMS = 20
LOG_NB_ATOMS = 20  # number of log atoms
LOG_THRESHOLD = 1.0  # where does the log start (1 for full log)
SPACING = 2

