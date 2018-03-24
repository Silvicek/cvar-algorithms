import time
from cliffwalker import *


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


def optimal_path(world, policy):
    """ Optimal deterministic path. """
    s = world.initial_state
    states = [s]
    t = Transition(s, 0, 0)
    while s not in world.goal_states:
        a = policy.next_action(t)
        t = max(world.transitions(s)[a], key=lambda t: t.prob)
        s = t.state
        if s in states:
            print("ERROR: path repeats {}, last action={}".format(s, world.ACTION_NAMES[a]))
            return states
        states.append(s)
    return states