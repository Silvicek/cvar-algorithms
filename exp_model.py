""" Standard policy iteration methods stored here. 
    Is not fully compatible with distributional setting.
"""
from cliffwalker import *
import numpy as np
from visual import show_fixed
from util import q_to_v_argmax


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
    Q = np.zeros((len(actions), H, W))
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
    Q = np.zeros((len(actions), H, W))
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


if __name__ == '__main__':
    # NOTE: PI doesn't converge with gamma=1
    # gamma = 0.99
    # Q = policy_iteration()

    Q = value_iteration()

    show_fixed(initial_state, q_to_v_argmax(Q), np.argmax(Q, axis=0))
