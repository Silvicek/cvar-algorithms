""" Used for generating various unrelated plots used in the thesis. """
import matplotlib.pyplot as plt
import scipy.stats
from cycler import cycler
# from cvar.gridworld.core.util import softmax

from cvar.gridworld.core.constants import *

# plt.rc('axes', prop_cycle=(cycler('color', ['#1f77b4', '#d62728'])))

# tex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# ====================


def pdf_to_cvar(prob, var, alpha):
    print(prob[:10]/alpha)
    print(np.sum(prob[:300]))
    print(sum(prob))
    p = 0.
    cv = 0.
    for p_, v_ in zip(prob, var):
        if p + p_ >= alpha:
            cv += (alpha - p) * v_
            break
        else:
            cv += p_ * v_
            p += p_
    return v_, cv / alpha


def plot_cvar_pdf(prob, vars, alpha, discrete=False):
    var, cvar = pdf_to_cvar(prob, vars, alpha)

    if discrete:
        n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75, edgecolor='black')
        print(n, bins, patches)
        plt.show()
    else:
        fig, ax = plt.subplots(1, figsize=(8,4))

        ax.plot(vars, prob)
        ax.vlines(x=var, ymin=0., ymax=0.001, linestyles='--', colors='g', label='$VaR_{%.2f}$' % alpha)
        ax.vlines(x=cvar, ymin=0., ymax=0.001, linestyles='--', colors='r', label='$CVaR_{%.2f}$' % alpha)

        # plt.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title('Probability Distribution Function')
        plt.grid()
        plt.legend()
        plt.show()


def cvar_multinomial():
    alpha = 0.05

    # student
    distribution = scipy.stats.t(1)
    vars = np.arange(-10, 10, 0.01)
    prob = distribution.pdf(vars) / 100

    # gaussian mixture
    d1 = scipy.stats.t(1, -2)
    d2 = scipy.stats.norm(5,1)
    d3 = scipy.stats.norm(-0,1)
    d4 = scipy.stats.norm(-7,0.5)
    vars = np.arange(-10, 10, 0.01)
    prob = (0.3 * d1.pdf(vars) + 0.3 * d2.pdf(vars) + 0.37 * d3.pdf(vars) + 0.03*d4.pdf(vars)) / 100
    # prob = ( d3.pdf(vars)) / 100


    # multinomial
    nb_atoms = 50
    atoms = softmax(np.random.random(nb_atoms))
    var_values = np.random.random([nb_atoms])*10 - 5
    var_values.sort()

    x = np.random.lognormal(sigma=0.6, size=10000) + np.random.randn(10000)

    # the histogram of the data
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75, edgecolor='black')
    # plt.show()

    plot_cvar_pdf(prob, vars, alpha)
    # plot_cvar_pdf(atoms, var_values, alpha, discrete=True)


def loss_functions():
    # fig, ax = plt.subplots(1,4)

    x = np.linspace(-1, 1, 201)
    print(x)


    y_mse = x ** 2

    # y_10 = np.where(x >= 0, 0.1*x, (0.1-1)*(-x))
    y_med = x * (0.5 - (x < 0))
    y_30 = x * (0.3 - (x < 0))
    y_70 = x * (0.7 - (x < 0))


    fig = plt.figure(figsize=(5,4))
    # y_90 = np.abs(x)
    plt.plot(x, y_mse)
    plt.plot(x, y_med)
    plt.plot(x, y_30)
    plt.plot(x, y_70)

    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([0, 1])

    plt.grid()

    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.legend(['MSE', '$\\alpha=0.5$', '$\\alpha=0.3$', '$\\alpha=0.7$'])

    plt.savefig('../data/plots/losses.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    loss_functions()











