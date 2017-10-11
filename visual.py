import matplotlib.pyplot as plt
import numpy as np
from cliffwalker import *


# gives a state-value function v(s) based on action-value function q(s, a) and policy
# for debugging and visualization
def q_to_v(Q, policy):  # TODO: move
    Vnew = np.zeros((H, W))
    for s in states():
        activity_probs = policy(s, Q)
        for a in actions:
            Vnew[s.y, s.x] += activity_probs[a] * Q[a, s.y, s.x]
    return Vnew


# visualizes the final value function with a fixed policy
def show_results(start_state, policy, Q):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = plt.gca()

    # darken cliff
    V = q_to_v(Q, policy)
    print(Q[0])
    print(Q[1])
    cool = np.min(V) * 1.1
    for s in cliff_states:
        V[s.y, s.x] = cool

    im = ax.imshow(V, interpolation='nearest', origin='upper')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    ax.text(start_state[1], start_state[0], 'S', ha='center', va='center', fontsize=20)
    for s in goal_states:
        ax.text(s[1], s[0], 'G', ha='center', va='center', fontsize=20)

    # arrows
    offsets = {0: (0.4, 0), 1: (-0.4, 0), 2: (0, 0.4), 3: (0, -0.4)}
    dirs = {0: (-0.8, 0), 1: (0.8, 0), 2: (0, -0.8), 3: (0, 0.8)}
    for s in states():
        if s in cliff_states:
            continue
        if s in goal_states:
            continue

        a = np.argmax(policy(s, Q))
        ax.add_patch(plt.Arrow(s.x + offsets[a][0], s.y + offsets[a][1], dirs[a][0], dirs[a][1], color='white'))

    plt.show()