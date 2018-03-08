import time
from util.constants import gamma
from cliffwalker import *
from plots.grid_plot_machine import PlotMachine
from policy_improvement.policies import VarBasedQPolicy
from q_learning import q_learning


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
    t = Transition(s, 0, 0)
    while s not in world.goal_states and i < max_iters:
        a = policy.next_action(t)
        A.append(a)

        if plot_machine is not None:
            plot_machine.step(s, a)
            time.sleep(0.5)

        t = world.sample_transition(s, a)

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


if __name__ == '__main__':

    world = GridWorld(4, 6, random_action_p=0.1)
    alpha = 0.25
    # =============== VI setup
    Q = q_learning(world, alpha)


    policy = VarBasedQPolicy(Q, alpha)

    # =============== plot dynamic
    V_visual = np.max(np.array([Q.Q[ix].expected_value() for ix in np.ndindex(Q.Q.shape)]).reshape(Q.Q.shape), axis=-1)

    # print(V_visual)
    plot_machine = PlotMachine(world, V_visual)
    # policy = var_policy
    for i in range(100):
        S, A, R = epoch(world, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()