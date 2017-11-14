from cliffwalker import *
from util import *
from policies import AlphaBasedPolicy, VarBasedPolicy, NaiveCvarPolicy, FixedPolicy, GreedyPolicy
from random_variable import RandomVariable, MIN_VALUE, MAX_VALUE
import numpy as np
from visual import show_fixed, PlotMachine
import time


np.random.seed(1337)
np.set_printoptions(3)


def cvar_from_samples(samples, alpha):
    samples = np.sort(samples)
    alpha_ix = int(np.round(alpha * len(samples)))
    var = samples[alpha_ix - 1]
    cvar = np.mean(samples[:alpha_ix])
    return var, cvar


def policy_iteration(world):
    Q = init_q(world)
    i = 0
    while True:
        expvals = expected_value(Q)
        Q_ = eval_fixed_policy(world, np.argmax(expvals, axis=0))

        if converged(Q, Q_) and i != 0:
            print("policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def naive_cvar_policy_iteration(world, alpha):
    Q = init_q(world)
    i = 0
    while True:
        cvars = cvar(Q, alpha)
        Q_ = eval_fixed_policy(np.argmax(cvars, axis=0))

        if converged(Q, Q_) and i != 0:
            print("naive cvar policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def init_q(world):
    Q = np.empty((4, world.height, world.width), dtype=object)
    for ix in np.ndindex(Q.shape):

        Q[ix] = RandomVariable()
    return Q


def eval_fixed_policy(world, P):
    Q = init_q(world)
    i = 0
    while True:
        Q_ = value_update(world, Q, P)
        if converged(Q, Q_) and i != 0:
            break
        Q = Q_
        i += 1

    return Q


def value_update(world, Q, P):
    """
    One value update step.
    :param Q: (A, M, N): current Q-values
    :param P: (M, N): indices of actions to be selected
    :return: (A, M, N): new Q-values
    """

    Q_ = init_q(world)
    for s in world.states():
        for a, action_transitions in zip(world.ACTIONS, world.transitions(s)):

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


def several_epochs(arg):
    np.random.seed()
    world, policy, nb_epochs = arg
    rewards = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        S, A, R = epoch(world, policy)
        policy.reset()
        rewards[i] = np.sum(R)

    return rewards


def policy_stats(world, policy, alpha, nb_epochs, verbose=True):
    import copy
    import multiprocessing as mp
    threads = 4

    with mp.Pool(threads) as p:
        rewards = p.map(several_epochs, [(world, copy.deepcopy(policy), int(nb_epochs/threads)) for _ in range(threads)])

    rewards = np.array(rewards).flatten()

    # clip to make the decision distribution more realistic
    rewards = np.clip(rewards, MIN_VALUE, MAX_VALUE)

    var, cvar = cvar_from_samples(rewards, alpha)
    if verbose:
        print('----------------')
        print(policy.__name__)
        print('expected value=', np.mean(rewards))
        print('cvar_{}={}'.format(alpha, cvar))
        # print('var_{}={}'.format(alpha, var))

    return cvar


def exhaustive_stats(world, epochs, *args):
    Q = policy_iteration(world)

    alphas = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    cvars = np.zeros((len(args), len(alphas)))
    names = []

    for i, policy in enumerate(args):
        names.append(policy.__name__)
        for j, alpha in enumerate(alphas):
            pol = policy(Q, alpha)

            cvars[i, j] = policy_stats(world, pol, alpha=alpha, nb_epochs=int(epochs), verbose=False)

            print('{}_{} done...'.format(pol.__name__, alpha))

    import pickle
    pickle.dump({'cvars': cvars, 'alphas': alphas, 'names': names}, open('files/stats.pkl', 'wb'))
    print(cvars)

    from visual import plot_cvars
    plot_cvars()


def epoch(world, policy, max_iters=100, plot_machine=None):
    """
    Evaluates a single epoch starting at start_state, using a given policy.
    :param start_state: 
    :param policy: Policy instance
    :param max_iters: end the epoch after this #steps
    :return: States, Actions, Rewards
    """
    s = world.initial_state
    S = [s]
    A = []
    R = []
    i = 0
    r = 0
    t = Transition(s, 0, 0)
    while s not in world.goal_states and i < max_iters:
        a = policy.next_action(t)

        if plot_machine is not None:
            plot_machine.step(s, a)
            time.sleep(0.5)


        A.append(a)
        trans = world.transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        t = trans[np.random.choice(len(trans), p=state_probs)]

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


def sample_runs(p, z):

    for i in range(2, 7):
        x = np.random.choice(z, size=(int(10**i)), p=p)
        print(len(x),)
        var, cvar = cvar_from_samples(x, alpha)
        print('1e{} - sample var: {}, cvar: {}'.format(i, var, cvar))


if __name__ == '__main__':

    # TODO: try naive PI

    world_ideal = GridWorld(4, 6, random_action_p=0.3)
    world_tweaked = GridWorld(4, 6, random_action_p=0.1)

    # =============== PI setup
    alpha = 0.4
    Q = policy_iteration(world_ideal)

    greedy_policy = GreedyPolicy(Q)
    naive_cvar_policy = NaiveCvarPolicy(Q, alpha=alpha)
    var_policy = VarBasedPolicy(Q, alpha=alpha)

    # exhaustive_stats(world_ideal, 1e6, GreedyPolicy, NaiveCvarPolicy, VarBasedPolicy)

    # =============== PI stats
    nb_epochs = 100000
    # policy_stats(world_ideal, greedy_policy, alpha, nb_epochs=nb_epochs)
    # policy_stats(world_ideal, var_policy, alpha, nb_epochs=nb_epochs)

    policy_stats(world_tweaked, greedy_policy, alpha, nb_epochs=nb_epochs)
    policy_stats(world_tweaked, var_policy, alpha, nb_epochs=nb_epochs)

    # policy_stats(world_ideal, naive_cvar_policy, alpha, nb_epochs=nb_epochs)

    # =============== plot fixed
    Q_exp = expected_value(Q)
    V_exp = q_to_v_argmax(world_ideal, Q_exp)
    Q_cvar = cvar(Q, alpha)
    V_cvar = q_to_v_argmax(world_ideal, Q_cvar)
    # show_fixed(initial_state, q_to_v_argmax(Q_exp), np.argmax(Q_exp, axis=0))
    # show_fixed(initial_state, q_to_v_argmax(Q_cvar), np.argmax(Q_cvar, axis=0))

    # =============== plot dynamic
    plot_machine = PlotMachine(world_ideal, V_exp)
    policy = var_policy
    policy = greedy_policy
    for i in range(100):
        S, A, R = epoch(world_ideal, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()

    # ============== other
    # Q_cvar = eval_fixed_policy(np.argmax(Q_cvar, axis=0))
    # interesting = Q_cvar[2, -1, 0]
    # print(interesting.cvar(alpha))
    # interesting.plot()



