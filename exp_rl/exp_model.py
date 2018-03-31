""" Standard RL methods stored here - VI, PI, Q-learning.
    Is not fully compatible with distributional setting.
"""

from cliffwalker import GridWorld
from plots.grid_plot_machine import show_fixed
from util.constants import gamma
import numpy as np

# ========================================
# algorithms
# ========================================


def value_iteration(world):
    Q = np.zeros((len(world.ACTIONS), world.height, world.width))
    i = 0
    while True:
        Q_ = value_update(world, Q, np.argmax(Q, axis=0))
        if converged(Q, Q_) and i != 0:
            print("value fully learned after %d iterations" % (i,))
            break
        Q = Q_
        i += 1
    return Q


def policy_iteration(world):
    Q = np.zeros((len(world.ACTIONS), world.height, world.width))
    i = 0
    while True:
        Q_ = eval_fixed_policy(np.argmax(Q, axis=0))
        print(i)
        if np.all(np.argmax(Q, axis=0) == np.argmax(Q_, axis=0)) and i != 0:
            print("policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def q_learning(world, max_episodes=1e3, max_iters=100):
    Q = np.zeros((len(world.ACTIONS), world.height, world.width))

    beta = 0.1  # learning rate
    eps = 0.5

    iter = 0
    while True:
        if iter % 100 == 0:
            beta = 1/max(1, iter/200)
            print("{}: beta={}".format(iter, beta))
        # ==========================
        s = world.initial_state

        i = 0
        while i < max_iters:
            # sample next action
            a = policy_sample(epsilon_greedy_policy(eps), s, Q)

            # sample next transition
            t = world.sample_transition(s, a)
            r, s_ = t.reward, t.state

            # update Q
            if s_ in world.goal_states:
                Q[a, s.y, s.x] = (1 - beta) * Q[a, s.y, s.x] + beta * r
                break
            else:
                a_ = np.argmax(Q[:, s_.y, s_.x])
                Q[a, s.y, s.x] = (1-beta)*Q[a, s.y, s.x] + beta*(r + gamma * Q[a_, s_.y, s_.x])

            s = s_

        # update learning parameters
        # if iter > 0.3*max_episodes:
        # eps = 1/(iter + 1)
        # print("{}: eps={}".format(iter, eps))


        iter += 1

        if iter > max_episodes:
            break

    return Q


# ========================================
# policies
# ========================================


# random policy: each action has the same probability
def random_policy(s, Q):
    return [1.0 / len(GridWorld.ACTIONS) for a in GridWorld.ACTIONS]


# greedy policy gives the best action (based on action value function Q) a probability of 1, others are given 0
def greedy_policy(s, Q):
    probs = np.zeros_like(GridWorld.ACTIONS, dtype=float)
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
    return np.random.choice(GridWorld.ACTIONS, p=p)

# ========================================
# other
# ========================================


def value_update(world, Q, P):
    """
    One value update step.
    :param Q: (A, M, N): current Q-values
    :param P: (M, N): indices of actions to be selected
    :return: (A, M, N): new Q-values
    """
    Q_ = np.array(Q)
    for s in world.states():
        for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):
            t_p = np.array([t.prob for t in action_transitions])

            t_q = [t.reward + gamma * Q[P[t.state.y, t.state.x], t.state.y, t.state.x] for t in action_transitions]
            Q_[a, s.y, s.x] = np.dot(t_p, t_q)

    return Q_


def converged(Q, Q_):
    return np.linalg.norm(Q-Q_)/Q.size < 1e-5


def eval_fixed_policy(world, P):
    Q = np.zeros((len(world.ACTIONS), world.height, world.width))
    i = 0
    while True:
        Q_ = value_update(world, Q, P)
        if converged(Q, Q_) and i != 0:
            break
        Q = Q_
        i += 1
    return Q


def q_to_v_argmax(world, Q):
    """ Converts Q function to V by choosing the best action. """
    Vnew = np.zeros((world.height, world.width))
    for s in world.states():
        a = np.argmax(Q[:, s.y, s.x])
        Vnew[s.y, s.x] = Q[a, s.y, s.x]
    return Vnew


if __name__ == '__main__':
    np.random.seed(2)
    world = GridWorld(40, 60, random_action_p=0.1)

    # Q = policy_iteration(world)
    # Q = value_iteration(world)
    Q = q_learning(world, max_episodes=50000)

    show_fixed(world, q_to_v_argmax(world, Q), np.argmax(Q, axis=0))
