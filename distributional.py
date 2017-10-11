from cliffwalker import *
import matplotlib.pyplot as plt
from visual import show_results
from exp_model import greedy_policy

REWARD_RANGE = (FALL_REWARD-10, 1)


class RandomVariable:

    def __init__(self, p=None, z=None):
        self.z = np.arange(*REWARD_RANGE) if z is None else np.copy(z)
        if p is None:
            self.p = np.zeros_like(self.z)
            self.p[-1] = 1.0
        else:
            self.p = np.copy(p)

    def expected_value(self):
        return np.dot(self.z, self.p)

    def __add__(self, r):
        # uses the fact that rewards are all negative ints
        # correct version: z += r
        if r >= 0:
            return RandomVariable(p=self.p)
        p = np.roll(self.p, r)
        p[0] += np.sum(p[r:])
        p[r:] = 0
        if abs(np.sum(p) - 1.0) > 0.001:
            print('PROBLEMS:{:.10f}'.format(np.sum(p)))
        return RandomVariable(p)

    def __mul__(self, gamma):
        return RandomVariable(z=gamma*self.z, p=self.p)

    def __str__(self):
        return 'p:{}\nz:{}'.format(self.p, self.z)

    def plot(self):
        ax = plt.gca()

        ax.bar(self.z, self.p, width=0.9, )

        ax.set_ylim([0., 1.1 * np.max(self.p)])
        plt.grid(True)

        plt.show()


# ===================== algorithms
def expected_value(rv):
    return rv.expected_value()

expected_value = np.vectorize(expected_value)


def policy_iteration():
    Q = init_q()
    i = 0
    while True:
        expvals = expected_value(Q)
        Q_ = eval_fixed_policy(np.argmax(expvals, axis=0))

        # if np.all(np.argmax(expected_value(Q), axis=0) == np.argmax(expected_value(Q_), axis=0)) and i != 0:
        if converged(Q, Q_) and i != 0:
            print("policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def init_q():
    Q = np.empty((4, H, W), dtype=object)
    for ix in np.ndindex(Q.shape):

        Q[ix] = RandomVariable()
    return Q


def eval_fixed_policy(P):
    Q = init_q()
    i = 0
    print('eval')
    while True:
        Q_ = value_update(Q, P)
        if converged(Q, Q_) and i != 0:
        # if i > 100:
            break
        Q = Q_
        i += 1

    # print(expected_value(Q))
    # show_results(initial_state, greedy_policy, expected_value(Q))
    return Q


def value_update(Q, P):
    """
    One value update step.
    :param Q: (A, M, N): current Q-values
    :param P: (M, N): indices of actions to be selected
    :return: (A, M, N): new Q-values
    """
    gamma = 1 # TODO: gamma != 1

    Q_ = init_q()
    for s in states():
        for a, action_transitions in zip(actions, transitions(s)):

            # transition probabilities
            t_p = np.array([t.prob for t in action_transitions])
            # random variables created by transitioning
            t_q = [Q[P[t.state.y, t.state.x], t.state.y, t.state.x] * gamma + t.reward for t in action_transitions]
            # picked out new probability vectors
            t_p_ = np.array([q.p for q in t_q])
            # weight the prob. vectors by transition probs
            # new_p = np.einsum('i,ij->ij', t_p, t_p_)
            new_p = np.matmul(t_p, t_p_)

            Q_[a, s.y, s.x] = RandomVariable(p=new_p)

    return Q_


def converged(Q, Q_):
    p = np.array([rv.p for rv in Q.flat])
    p_ = np.array([rv.p for rv in Q_.flat])

    return np.linalg.norm(p-p_)/Q.size < 0.001


# gives a state-value function v(s) based on action-value function q(s, a) and policy
# for debugging and visualization
def q_to_v(Q, policy):
    Vnew = np.zeros((H, W))
    for s in states():
        activity_probs = policy(s, Q)
        for a in actions:
            Vnew[s.y, s.x] += activity_probs[a] * Q[a, s.y, s.x]
    return Vnew


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
    Q = policy_iteration()
    show_results(initial_state, greedy_policy, expected_value(Q))
