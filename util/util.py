import numpy as np

# def clip(ix):
#     new_ix = max(0, min(MAX_VALUE - MIN_VALUE, ix))
#     return new_ix


def spaced_atoms(nb_atoms, spacing, log_atoms, log_threshold):
    assert log_atoms <= nb_atoms
    assert spacing > 1

    if log_atoms != 0:
        lin = np.linspace(log_threshold, 1, nb_atoms - log_atoms)
        if spacing < 2:
            log = np.array([0, log_threshold * 0.5 / spacing ** log_atoms] + [log_threshold / spacing ** (log_atoms - i)
                                                                              for i in range(log_atoms-1)])
        else:
            log = np.array([0] + [log_threshold / spacing ** (log_atoms - i) for i in range(log_atoms)])

        atoms = np.hstack((log, lin))
    else:
        atoms = np.linspace(0, 1, nb_atoms+1)

    assert np.all(atoms == np.array(sorted(atoms)))

    return atoms


def softmax(x):
    exp = np.exp(x)
    if len(x.shape) > 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.sum(exp)

