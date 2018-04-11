import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_cvars():
    """ Old function to plot policy improvement comparisons. """
    import pickle

    data = pickle.load(open('data/stats.pkl', 'rb'))

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

