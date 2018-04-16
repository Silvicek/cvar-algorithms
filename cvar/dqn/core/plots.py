import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cvar.gridworld.core.cvar_computation import yc_to_var


class PlotMachine:

    def __init__(self, nb_atoms, nb_actions, action_set=None):

        # TODO: unify linear atoms
        self.atoms = np.arange(0, nb_atoms + 1) / nb_atoms

        self.limits = None

        self.fig, self.ax = plt.subplots(1, 2)
        self.fig.canvas.draw()

        self.dist_plot = [self.ax[0].step(np.insert(self.atoms, -1, 1), np.zeros(nb_atoms+2))[0] for _ in range(nb_actions)]
        self.var_plot = [self.ax[1].step(np.insert(self.atoms, -1, 1), np.zeros(nb_atoms+2))[0] for _ in range(nb_actions)]

        if action_set is not None:
            plt.legend(action_set, loc='upper left')

        self.sess = tf.get_default_session()
        self.act_cvar = tf.get_default_graph().get_tensor_by_name("cvar_dqn/out_func/cvar/out:0")
        self.act_var = tf.get_default_graph().get_tensor_by_name("cvar_dqn/out_func/var/out:0")

    def plot_distribution(self, obs):
        # TODO: var/cvar
        cvar_out = self.sess.run(self.act_cvar, {"cvar_dqn/observation:0": obs})[0]
        var_out = self.sess.run(self.act_var, {"cvar_dqn/observation:0": obs})[0]
        dist_out = [yc_to_var(self.atoms, cvar_out[a]) for a in range(len(cvar_out))]

        if self.limits is None:
            self.limits = [np.min(dist_out), np.max(var_out)]
        else:
            self.limits = [min(np.min(dist_out), self.limits[0]), max(np.max(var_out), self.limits[1])]

        self.ax[0].set_ylim(self.limits)
        self.ax[1].set_ylim(self.limits)

        for plot, values in zip([self.dist_plot, self.var_plot], [dist_out, var_out]):
            for line, quant in zip(plot, values):
                y_data = np.zeros(len(quant)+2)
                y_data[1:-1] = quant
                y_data[0] = self.limits[0]
                y_data[-1] = self.limits[-1]
                line.set_ydata(y_data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-10)

