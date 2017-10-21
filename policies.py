from util import *


class Policy:
    """ Abstract class representing different policies. """

    __name__ = 'Policy'

    def next_action(self, t):
        raise NotImplementedError()

    def reset(self):
        pass


class FixedPolicy(Policy):
    __name__ = 'Fixed'

    def __init__(self, P, alpha=None):
        self.P = P

    def next_action(self, t):
        return self.P[t.state.y, t.state.x]


class GreedyPolicy(Policy):
    __name__ = 'Greedy'

    def __init__(self, Q, alpha=None):
        self.Q = Q

    def next_action(self, t):
        s = t.state
        return np.argmax(expected_value(self.Q[:, s.y, s.x]))


class NaiveCvarPolicy(Policy):
    __name__ = 'Naive CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.alpha = alpha

    def next_action(self, t):
        s = t.state
        action_distributions = self.Q[:, s.y, s.x]
        a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        return a


class AlphaBasedPolicy(Policy):
    __name__ = 'alpha-based CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.init_alpha = alpha
        self.alpha = alpha
        self.s_old = None
        self.a_old = None

    def next_action(self, t):
        s = t.state

        action_distributions = self.Q[:, s.y, s.x]
        old_action = np.argmax(expected_value(action_distributions))

        if self.alpha > 0.999:
            return old_action

        if self.s_old is not None:
            self.update_alpha(self.s_old, self.a_old, t)
        a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        self.s_old = s
        self.a_old = a

        return a

    def update_alpha(self, s, a, t):
        """
        Correctly updates next alpha with discrete variables.
        :param s: state we came from
        :param a: action we took
        :param t: transition we sampled
        """
        s_ = t.state
        s_dist = self.Q[a, s.y, s.x]

        a_ = np.argmax(expected_value(self.Q[:, s_.y, s_.x]))

        s__dist = self.Q[a_, s_.y, s_.x]

        var_ix = s_dist.var_index(self.alpha)

        var__ix = clip(var_ix - t.reward)

        # prob before var
        p_pre = np.sum(s_dist.p[:var_ix])
        # prob at var
        p_var = s_dist.p[var_ix]
        # prob before next var
        p__pre = np.sum(s__dist.p[:var__ix])
        # prob at next var
        p__var = s__dist.p[var__ix]

        # how much does t add to the full var
        p_portion = (t.prob * p__var) / self.p_portion_sum(s, a, var_ix)

        # we care about this portion of var
        p_active = (self.alpha - p_pre) / p_var

        self.alpha = p__pre + p_active * p__var * p_portion

    def p_portion_sum(self, s, a, var_ix):

        p_portion = 0.

        for t_ in transitions(s)[a]:
            action_distributions = self.Q[:, t_.state.y, t_.state.x]
            a_ = np.argmax(expected_value(action_distributions))
            p_portion += t_.prob*action_distributions[a_].p[clip(var_ix - t_.reward)]

        return p_portion

    def reset(self):
        self.alpha = self.init_alpha
        self.s_old = None
        self.a_old = None


class VarBasedPolicy(Policy):
    __name__ = 'VaR-based CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.alpha = alpha
        self.var = None

    def next_action(self, t):

        action_distributions = self.Q[:, t.state.y, t.state.x]

        if self.var is None:
            cvars = cvar(action_distributions, self.alpha)
            a = np.argmax(cvars)
            self.var = action_distributions[a].var(self.alpha)
        else:
            self.var = np.clip((self.var - t.reward) / gamma, MIN_VALUE, MAX_VALUE)

        a = np.argmax([d.exp_(self.var) for d in action_distributions])

        return a

    def reset(self):
        self.var = None

