import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# arrows
offsets = {0: (0.4, 0), 1: (-0.4, 0), 2: (0, 0.4), 3: (0, -0.4)}
dirs = {0: (-0.8, 0), 1: (0.8, 0), 2: (0, -0.8), 3: (0, 0.8)}


class PlotMachine:

    def __init__(self, world, V):
        self.V = V
        # darken cliff
        cool = np.min(V) * 1.1
        for s in world.cliff_states:
            V[s.y, s.x] = cool

        plt.ion()

        self.fig, self.ax = plt.subplots()

        im = self.ax.imshow(V, interpolation='nearest', origin='upper')
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='off', right='off', left='off', labelleft='off')
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        self.ax.text(world.initial_state.x, world.initial_state.y, 'S', ha='center', va='center', fontsize=20)
        for s in world.goal_states:
            self.ax.text(s[1], s[0], 'G', ha='center', va='center', fontsize=20)
        for s in world.risky_goal_states:
            self.ax.text(s[1], s[0], 'R', ha='center', va='center', fontsize=20)

        self.arrow = self.ax.add_patch(plt.Arrow(0, 0, 1, 1, color='white'))

    def step(self, s, a):

        self.arrow.remove()
        arrow = plt.Arrow(s.x + offsets[a][0], s.y + offsets[a][1], dirs[a][0], dirs[a][1], color='white')
        self.arrow = self.ax.add_patch(arrow)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def plot_cvars():
    import pickle

    data = pickle.load(open('files/stats.pkl', 'rb'))

    cvars = data['cvars']
    alphas = np.tile(data['alphas'], (len(cvars), 1))
    ax = plt.gca()
    ax.plot(alphas.T, cvars.T, '-')
    ax.set_xscale('log')
    ax.set_xticks(alphas[0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.invert_xaxis()
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('CVaR$_\\alpha$')
    # ax.set_ylim([-50, -10])
    ax.legend(data['names'])
    ax.grid()
    plt.show()


# visualizes the final value function with a fixed policy
def show_fixed(world, V, P):

    ax = plt.gca()

    # darken cliff
    cool = np.min(V) * 1.1
    for s in world.cliff_states:
        V[s.y, s.x] = cool

    im = ax.imshow(V, interpolation='nearest', origin='upper')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    ax.text(world.initial_state[1], world.initial_state[0], 'S', ha='center', va='center', fontsize=20)
    for s in world.goal_states:
        ax.text(s[1], s[0], 'G', ha='center', va='center', fontsize=20)
    for s in world.risky_goal_states:
        ax.text(s[1], s[0], 'R', ha='center', va='center', fontsize=20)

    for s in world.states():
        if s in world.cliff_states:
            continue
        if s in world.goal_states:
            continue
        if s in world.risky_goal_states:
            continue

        a = P[s.y, s.x]
        ax.add_patch(plt.Arrow(s.x + offsets[a][0], s.y + offsets[a][1], dirs[a][0], dirs[a][1], color='white'))

    plt.show()


if __name__ == '__main__':

    plot_cvars()
