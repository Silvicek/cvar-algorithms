import numpy as np
from pulp import *
import matplotlib.pyplot as plt
np.random.seed(1337)

def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)


nb_atoms = 3
nb_transitions = 2

transition_p = np.array([0.25, 0.75])

atoms = np.array([0., 0.25, 0.5, 1.])
atom_p = atoms[1:] - atoms[:-1]

var_values = np.array([[-1, 0, 0.5],
                      [-3, -2, -1]])
# var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
# var_values.sort()

# transition_p = softmax(np.random.random(nb_transitions))
# atoms = np.zeros(nb_atoms+1)
# atoms[1:] = np.cumsum(softmax(np.random.random(nb_atoms)))
# atom_p = atoms[1:] - atoms[:-1]
#
# var_values = np.random.randint(-10, 10, [nb_transitions, nb_atoms])
# var_values.sort()

print(atoms)
print(atom_p)
print(var_values)


def tamar_lp():
    pass


def wasserstein():
    pass


def simple_sort():
    # 0) weight by transition probs
    p = np.outer(transition_p, atom_p).flatten()

    # 1) sort
    sortargs = var_values.flatten().argsort()
    var_sorted = var_values.flatten()[sortargs]
    p_sorted = p.flatten()[sortargs]

    # 2) compute yV for each atom

    yV = np.zeros(nb_atoms)
    for ix, atom in enumerate(atoms[1:]):
        cs = 0.
        cp = 0.
        for p_, v_ in zip(p_sorted, var_sorted):
            cp += p_
            if cp >= atom:
                p__ = atom - (cp-p_)
                cs += p__ * v_
                break
            else:
                cs += p_ * v_
        yV[ix] = cs

    print('--------------')
    print(yV)
    print(-11/16, -17/16, -21/16)
    # 3) get vars from yV
    last = 0.
    var_solution = np.zeros_like(yV)
    for i in range(nb_atoms):
        ddalpha = (yV[i]-last)/atom_p[i]
        var_solution[i] = ddalpha
        last = yV[i]

    return var_solution




print(simple_sort())




