import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# arrows
offsets = {0: (0.4, 0), 1: (-0.4, 0), 2: (0, 0.4), 3: (0, -0.4)}
dirs = {0: (-0.8, 0), 1: (0.8, 0), 2: (0, -0.8), 3: (0, 0.8)}


class PlotMachine:

    def __init__(self, world, V=None):
        if V is None:
            self.V = -1 * np.ones((world.height, world.width))
        else:
            self.V = V
        # darken cliff
        cool = np.min(self.V) * 1.1
        for s in world.cliff_states:
            self.V[s.y, s.x] = cool

        plt.ion()

        self.fig, self.ax = plt.subplots()

        im = self.ax.imshow(self.V, interpolation='nearest', origin='upper')
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

# TODO: unify imshow
class InteractivePlotMachine:

    def __init__(self, world, V):
        self.V = V
        img = np.array([V.V[ix].expected_value() for ix in np.ndindex(V.V.shape)]).reshape(V.V.shape)
        print(img.shape)

        self.fig, self.ax = grid_plot(world, img)
        self.fig.canvas.mpl_connect('button_press_event', self.handle_click)

        self.state_fig = None
        self.state_ax = None

    def handle_click(self, event):

        if event.xdata is None:
            return
        x, y = self._canvas_to_grid(event.xdata, event.ydata)

        if self.state_fig is None:
            self.state_fig, self.state_ax = plt.subplots(1, 2)

        # clear axes
        for ax in self.state_ax:
            ax.clear()

        self.V.V[y, x].plot(figax=(self.state_fig, self.state_ax))

    def _canvas_to_grid(self, xd, yd):
        offset = -0.5
        cell_length = 1
        x = int((xd - offset) / cell_length)
        y = int((yd - offset) / cell_length)
        return x, y



    def show(self):

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


def grid_plot(world, img=None):

    if img is None:
        img = -1 * np.ones((world.height, world.width))
    # darken cliff
    cool = np.min(img) * 1.1
    for s in world.cliff_states:
        img[s.y, s.x] = cool

    fig, ax = plt.subplots()

    im = ax.imshow(img, interpolation='nearest', origin='upper')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.text(world.initial_state.x, world.initial_state.y, 'S', ha='center', va='center', fontsize=20)
    for s in world.goal_states:
        ax.text(s[1], s[0], 'G', ha='center', va='center', fontsize=20)
    for s in world.risky_goal_states:
        ax.text(s[1], s[0], 'R', ha='center', va='center', fontsize=20)
    return fig, ax

