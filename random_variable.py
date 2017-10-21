import matplotlib.pyplot as plt
import numpy as np
from constants import *


class RandomVariable:

    def __init__(self, p=None, z=None):
        self.z = np.arange(MIN_VALUE, MAX_VALUE+1) if z is None else np.copy(z)
        if p is None:
            self.p = np.zeros_like(self.z)
            zero_ix, = np.where(self.z == 0)
            self.p[zero_ix] = 1.0
        else:
            self.p = np.copy(p)
        assert np.abs(np.sum(self.p) - 1.0) < 0.001

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

    def var_index(self, alpha):
        if alpha > 0.999:
            return len(self.z)-1
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1

        return i-1

    def exp_(self, s, alpha=None):
        if alpha is None:
            return np.dot(self.p, np.clip(self.z-s, None, 0))
        return 1./alpha*np.dot(self.p, np.clip(self.z-s, None, 0)) + s

    def __add__(self, r):
        # uses the fact that rewards are all integers
        # correct version: z += r
        assert abs(r) < MAX_VALUE - MIN_VALUE  # increase the range
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

        return RandomVariable(p=p)

    def __mul__(self, gamma):
        assert gamma == 1  # other values not supported
        return RandomVariable(z=gamma*self.z, p=self.p)

    def __str__(self):
        return 'p:{}\nz:{}'.format(self.p, self.z)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar(self.z, self.p, width=0.9, )

        ax.set_ylim([0., 1.1 * np.max(self.p)])
        ax.grid()

        fig.show()

    def plot_cdf(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(self.z, np.cumsum(self.p), where='post')
        ax.grid()
        plt.show()
