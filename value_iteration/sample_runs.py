import time
from util.constants import gamma
from util.runs import epoch
from cliffwalker import *
from plots.grid_plot_machine import PlotMachine
from policy_improvement.policies import GreedyPolicy, TamarPolicy, TamarVarBasedPolicy
from value_iteration import value_iteration
from util import cvar_computation


def several_epochs(arg):
    np.random.seed()
    world, policy, nb_epochs = arg
    rewards = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        S, A, R = epoch(world, policy)
        policy.reset()
        rewards[i] = np.sum(R)
        rewards[i] = np.dot(R, np.array([gamma ** i for i in range(len(R))]))

    return rewards


def policy_stats(world, policy, alpha, nb_epochs, verbose=True):
    import copy
    import multiprocessing as mp
    threads = 4

    with mp.Pool(threads) as p:
        rewards = p.map(several_epochs, [(world, copy.deepcopy(policy), int(nb_epochs/threads)) for _ in range(threads)])

    rewards = np.array(rewards).flatten()

    var, cvar = cvar_computation.v_c_from_samples(rewards, alpha)
    if verbose:
        print('----------------')
        print(policy.__name__)
        print('expected value=', np.mean(rewards))
        print('cvar_{}={}'.format(alpha, cvar))
        # print('var_{}={}'.format(alpha, var))

    return cvar, rewards


def exhaustive_stats(world, epochs, *args):
    V = value_iteration(world)

    alphas = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    cvars = np.zeros((len(args), len(alphas)))
    names = []

    for i, policy in enumerate(args):
        names.append(policy.__name__)
        for j, alpha in enumerate(alphas):
            pol = policy(V, alpha)

            cvars[i, j], _ = policy_stats(world, pol, alpha=alpha, nb_epochs=int(epochs), verbose=False)

            print('{}_{} done...'.format(pol.__name__, alpha))

    import pickle
    pickle.dump({'cvars': cvars, 'alphas': alphas, 'names': names}, open('files/stats.pkl', 'wb'))
    print(cvars)

    from plots.other import plot_cvars
    plot_cvars()


if __name__ == '__main__':

    world = GridWorld(4, 6, random_action_p=0.1)

    # =============== VI setup
    V = value_iteration(world, max_iters=100)
    alpha = 0.9
    print(V.V[3, 0].y_cvar(alpha)/alpha)

    var_policy = TamarVarBasedPolicy(V, alpha)
    tamar_policy = TamarPolicy(V, alpha)

    # =============== VI stats
    # nb_epochs = int(1e6)
    # rewards_sample = []
    # for alpha in [0.1, 0.25, 0.5, 1.]:
    #     _, rewards = policy_stats(world, TamarPolicy(V, alpha), alpha, nb_epochs=nb_epochs)
    #     rewards_sample.append(rewards)
    # np.save('files/sample_rewards_tamar.npy', np.array(rewards_sample))
    # policy_stats(world, var_policy, alpha, nb_epochs=nb_epochs)

    # =============== plot dynamic
    V_visual = np.array([[V.V[i, j].y_cvar(alpha) for j in range(len(V.V[i]))] for i in range(len(V.V))])
    # print(V_visual)
    plot_machine = PlotMachine(world, V_visual)
    policy = tamar_policy
    # policy = var_policy
    for i in range(100):
        S, A, R = epoch(world, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()