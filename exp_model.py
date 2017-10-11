from cliffwalker import *
import matplotlib.pyplot as plt


# random policy: each action has the same probability
def random_policy(s, Q):
    return [1.0 / len(actions) for a in actions]


# greedy policy gives the best action (based on action value function Q) a probability of 1, others are given 0
def greedy_policy(s, Q):
    probs = np.zeros_like(actions, dtype=float)
    probs[np.argmax(Q[:, s.y, s.x])] = 1.0
    return probs


# epsilon-greedy policy gives random action with probability eps or greedy one otherwise
def epsilon_greedy_policy(eps=0.0):
    def epsilon_greedy_policy_helper(s, Q):
        if np.random.uniform() < eps:
            return random_policy(s, Q)
        else:
            return greedy_policy(s, Q)

    return epsilon_greedy_policy_helper


def policy_sample(policy, *args):
    p = policy(*args)
    return np.random.choice(actions, p=p)

# ===================== algorithms


def value_iteration():
    Q = np.zeros((len(actions), H, W))
    i = 0
    while True:
        Q_ = value_update(Q, np.argmax(Q, axis=0))
        if converged(Q, Q_) and i != 0:
            print("value fully learned after %d iterations" % (i,))
            break
        Q = Q_
        i += 1
    return Q


def policy_iteration():
    Q = np.zeros((4, H, W))
    i = 0
    while True:
        Q_ = eval_fixed_policy(np.argmax(Q, axis=0))

        if np.all(np.argmax(Q, axis=0) == np.argmax(Q_, axis=0)) and i != 0:
            print("policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


# ==================== other

def value_update(Q, P):
    """
    One value update step.
    :param Q: (A, M, N): current Q-values
    :param P: (M, N): indices of actions to be selected
    :return: (A, M, N): new Q-values
    """
    Q_ = np.array(Q)
    for s in states():
        for a, action_transitions in zip(actions, transitions(s)):
            # TODO: more effective
            t_p = np.array([t.prob for t in action_transitions])

            t_q = [t.reward + gamma * Q[P[t.state.y, t.state.x], t.state.y, t.state.x] for t in action_transitions]
            Q_[a, s.y, s.x] = np.dot(t_p, t_q)

    return Q_


def converged(Q, Q_):
    return np.linalg.norm(Q-Q_)/Q.size < 0.001


def eval_fixed_policy(P):
    Q = np.zeros((4, H, W))
    i = 0
    while True:
        Q_ = value_update(Q, P)
        if converged(Q, Q_) and i != 0:
            break
        Q = Q_
        i += 1
    return Q


# evaluates a single epoch starting at start_state, using a policy which can use
# an action-value function Q as a parameter
# returns a triple: states visited, actions taken, rewards taken
def epoch(start_state, policy, Q, max_iters=100):
    s = start_state
    S = [s]
    A = []
    R = []
    i = 0
    while s not in goal_states and i < max_iters:
        a = policy_sample(policy, s, Q)
        A.append(a)
        trans = transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        t = trans[np.random.choice(len(trans), p=state_probs)]

        R.append(t.reward)
        S.append(t.state)
        s = t.state
        i += 1

    return S, A, R


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


if __name__ == '__main__':
    Q = value_iteration()
    # Q = policy_iteration()
    show_results(initial_state, greedy_policy, Q)
