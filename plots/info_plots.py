import pickle
import numpy as np
import plots.grid
import matplotlib
import matplotlib.pyplot as plt
from q_learning.q_learning import ActionValueFunction, MarkovState

# ============================= FONTS
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 8})

# ============================= LOAD
world, Q = pickle.load(open('../files/models/q_10_15.pkl', 'rb'))


# ============================= Optimal path grids
alphas = [0.1, 0.25, 0.5, 1.]
fig, axs = plt.subplots(2, 2, figsize=(8.5, 5))

for ax, alpha in zip(axs.flatten(), alphas):
    img = np.max(np.array([Q.Q[ix].yc_alpha(alpha)/alpha for ix in np.ndindex(Q.Q.shape)]).reshape(Q.Q.shape), axis=-1)
    plots.grid.grid_plot(world, img=img, figax=(fig, ax), sg_size=10)

    path = Q.optimal_path(alpha)
    print(path)
    ax.plot([s[1] for s in path], [s[0] for s in path], '--', color='white')

    ax.set_title("$\\alpha={}$".format(alpha))
    ax.axis('off')

# plt.savefig('../files/plots/q_optimal_paths.pdf', bbox_inches='tight')
plt.show()


# # ============================= OPTIMAL PATHS
# print('ATOMS:', Q.atoms)
#
# for alpha in np.arange(0.05, 1, 0.05):
#     print(alpha)
#     pm = plots.grid.InteractivePlotMachine(world, Q, alpha=alpha, action_value=True)
#     pm.show()

