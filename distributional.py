from cliffwalker import *
import matplotlib.pyplot as plt
from visual import show_results
import exp_model


MIN_VALUE = FALL_REWARD-50
MAX_VALUE = 100


class RandomVariable:

    def __init__(self, p=None, z=None):
        self.z = np.arange(MIN_VALUE, MAX_VALUE+1) if z is None else np.copy(z)
        if p is None:
            self.p = np.zeros_like(self.z)
            zero_ix, = np.where(self.z == 0)
            self.p[zero_ix] = 1.0
        else:
            self.p = np.copy(p)

    def expected_value(self):
        return np.dot(self.z, self.p)

    def cvar(self, alpha):
        if alpha > 0.999:
            return self.expected_value()
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1
        i -= 1
        p = p - self.p[i]
        p_rest = alpha - p
        cvar = (np.dot(self.p[:i], self.z[:i]) + p_rest * self.z[i]) / alpha

        return cvar

    def var(self, alpha):
        if alpha > 0.999:
            return self.z[-1]
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1

        if p == alpha:
            var = self.z[i-1]
        else:
            var = self.z[i]

        return var

    def alpha_from_var(self, var):
        # TODO: deal with discrete distributions
        i, = np.where(self.z == var)
        # print(var, i, self.z)
        alpha = np.sum(self.p[:i[0]+1])

        return alpha

    def __add__(self, r):
        # uses the fact that rewards are all negative ints
        # correct version: z += r
        if r == 0:
            p = self.p
        elif r > 0:
            p = np.roll(self.p, r)
            p[-1] += np.sum(p[:r])
            p[:r] = 0
        else:
            p = np.roll(self.p, r)
            p[0] += np.sum(p[r:])
            p[r:] = 0
            if abs(np.sum(p) - 1.0) > 0.001:
                print('PROBLEMS:{:.10f}'.format(np.sum(p)))
        return RandomVariable(p=p)

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


def cvar_from_samples(samples, alpha):
    samples = np.sort(samples)
    alpha_ix = int(np.round(alpha * len(samples)))
    var = samples[alpha_ix - 1]
    cvar = np.mean(samples[:alpha_ix])
    return var, cvar


class Policy:
    """ Abstract class representing different policies. """

    __name__ = 'Policy'

    def next_action(self, s, r):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class GreedyPolicy(Policy):
    __name__ = 'Greedy'

    def __init__(self, Q, alpha):
        self.Q = Q

    def next_action(self, s, r):
        return np.argmax(expected_value(self.Q[:, s.y, s.x]))

    def reset(self):
        pass


class NaiveCvarPolicy(Policy):
    __name__ = 'Naive CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.alpha = alpha

    def reset(self):
        pass

    def next_action(self, s, r):
        action_distributions = self.Q[:, s.y, s.x]
        a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        return a


class AlphaBasedPolicy(Policy):
    __name__ = 'alpha-based CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.init_alpha = alpha
        self.alpha = alpha
        self.var = None

    def next_action(self, s, r):
        action_distributions = self.Q[:, s.y, s.x]
        old_action = np.argmax(expected_value(action_distributions))

        if self.alpha > 0.999:
            return old_action

        if self.var is not None:
            self.alpha = action_distributions[old_action].alpha_from_var(min(MAX_VALUE, (self.var - r)/gamma))

        # XXX: same cvar => problem
        # TODO: deal with this?
        cvars = np.array([d.cvar(self.alpha) for d in action_distributions])
        if len(np.unique(cvars)) == 1:
            a = old_action
        else:
            a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        self.var = action_distributions[a].var(self.alpha)

        # print('s: {}, r: {}, alpha: {}, var:{}'.format(s, r, self.alpha, self.var))
        # self.Q[a, s.y, s.x].plot()

        # return np.argmax(expected_value(Q[:, s.y, s.x]))
        return a

    def reset(self):
        self.alpha = self.init_alpha
        self.var = None


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


def policy_stats(policy, alpha, nb_epochs=1000, verbose=True):

    rewards = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        S, A, R = epoch(initial_state, policy)
        policy.reset()
        rewards[i] = np.sum(R)

    var, cvar = cvar_from_samples(rewards, alpha)
    if verbose:
        print('----------------')
        print(policy.__name__)
        print('expected value=', np.mean(rewards))
        print('cvar_{}={}'.format(alpha, cvar))
        print('----------------')

    return cvar


def exhaustive_stats(*args):
    Q = policy_iteration()

    alphas = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    cvars = np.zeros((len(args), len(alphas)))
    names = []

    for i, policy in enumerate(args):
        names.append(policy.__name__)
        for j, alpha in enumerate(alphas):
            pol = policy(Q, alpha)

            cvars[i, j] = policy_stats(pol, alpha=alpha, verbose=False)

            print('{}_{} done...'.format(pol.__name__, alpha))

    import pickle
    pickle.dump({'cvars': cvars, 'alphas': alphas, 'names': names}, open('stats.pkl', 'wb'))
    print(cvars)




# evaluates a single epoch starting at start_state, using a policy which can use
# an action-value function Q as a parameter
# returns a triple: states visited, actions taken, rewards taken
def epoch(start_state, policy, max_iters=100):
    s = start_state
    S = [s]
    A = []
    R = []
    i = 0
    r = 0
    while s not in goal_states and i < max_iters:
        a = policy.next_action(s, r)
        A.append(a)
        trans = transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        t = trans[np.random.choice(len(trans), p=state_probs)]

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


if __name__ == '__main__':

    # exhaustive_stats(GreedyPolicy, AlphaBasedPolicy, NaiveCvarPolicy)

    Q = policy_iteration()

    alpha = 0.1
    greedy_policy = GreedyPolicy(Q, alpha)
    alpha_policy = AlphaBasedPolicy(Q, alpha=alpha)
    naive_cvar_policy = NaiveCvarPolicy(Q, alpha=alpha)

    policy_stats(greedy_policy, alpha)
    policy_stats(alpha_policy, alpha)
    policy_stats(naive_cvar_policy, alpha)

    # show_results(initial_state, exp_model.greedy_policy, expected_value(Q))
