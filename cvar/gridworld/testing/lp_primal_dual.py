from pulp import *
import numpy as np
from util import cvar_computation


atoms = np.array([0, 1/4, 1/2, 1])

transition_p = np.array([0.25, 0.75])

yc_values = np.array([[-0.25, -0.25, 0], [-3/4, -5/4, -7/4]])

# atoms = np.array([0, 1/4, 1/2, 1])
#
# transition_p = np.array([1.])
#
# yc_values = np.array([[-3/4, -5/4, -7/4]])


t_atoms = np.array([atoms, atoms])

alpha = 1
nb_transitions = len(transition_p)
nb_atoms = len(atoms)-1


# ===========================================
# primal
# ===========================================
Xi = [LpVariable('xi_' + str(i)) for i in range(nb_transitions)]
I = [LpVariable('I_' + str(i)) for i in range(nb_transitions)]

prob = LpProblem(name='tamar')

for xi in Xi:
    prob.addConstraint(0 <= xi)
    prob.addConstraint(xi <= 1./alpha)
prob.addConstraint(sum([xi*p for xi, p in zip(Xi, transition_p)]) == 1)

for xi, i, yc, atoms in zip(Xi, I, yc_values, t_atoms):
    last_yc = 0.
    f_params = []
    atom_p = atoms[1:] - atoms[:-1]
    for ix in range(len(yc)):
        # linear interpolation as a solution to 'y = kx + q'
        k = (yc[ix]-last_yc)/atom_p[ix]

        q = last_yc - k * atoms[ix]
        prob.addConstraint(i >= k * xi * alpha + q)
        f_params.append((k, q))
        last_yc = yc[ix]

# opt criterion
prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

prob.solve()

print(prob.objective, '=', value(prob.objective))


# ===========================================
# dual
# ===========================================
var_values = [cvar_computation.yc_to_var(atoms, yc_values[i]) for i in range(nb_transitions)]

La = LpVariable('l_a')
Lb = [LpVariable('l_' + str(i) + '_b') for i in range(nb_transitions)]

Lij = [[LpVariable('l_' + str(i) + '_' + str(j)) for j in range(nb_atoms)] for i in range(nb_transitions)]

prob = LpProblem(name='dual', sense=LpMaximize)


for i in range(nb_transitions):

    prob.addConstraint(Lb[i] <= 0)

    for j in range(nb_atoms):
        prob.addConstraint(Lij[i][j] <= 0)
    prob.addConstraint(sum(Lij[i]) == -transition_p[i])

    prob.addConstraint(transition_p[i]*La + Lb[i] +
                       sum([Lij[i][j]*alpha*var_values[i][j] for j in range(nb_atoms)]) <= 0)


lb = sum([Lb[i]*1/alpha for i in range(nb_transitions)])
lij = sum([sum([Lij[i][j]*(atoms[j+1]*var_values[i][j]-yc_values[i][j]) for j in range(nb_atoms)])
           for i in range(nb_transitions)])

prob.setObjective(La+lb+lij)

prob.solve()

print(prob.objective, '=', value(prob.objective))


print(La.name, '=', value(La))
for i in range(nb_transitions):

    for v in Lb:
        print(v.name, '=', value(v))

    for v in Lij[i]:
        print(v.name, '=', value(v))














