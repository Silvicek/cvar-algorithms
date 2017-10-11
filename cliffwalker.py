import numpy as np
from itertools import chain
from collections import namedtuple

np.random.seed(1337)


# helper data structures:
# a state is given by row and column positions designated (y, x)
State = namedtuple('State', ['y', 'x'])

# encapsulates a transition to state and its probability
Transition = namedtuple('Transition', ['state', 'prob', 'reward'])  # transition to state with probability prob

# height and width of the gridworld
H, W = 2, 3

# available actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3
actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]

# the world will make a different action with probability
RANDOM_ACTION_P = 0.00

FALL_REWARD = -10

# initial state for debugging
initial_state = State(H-1, 0)

# set of goal states
goal_states = {State(H-1, W-1)}

# set of cliff states
cliff_states = {State(H-1, i) for i in range(1, W - 1)}

# undiscounted rewards
gamma = 1.


# iterator over all possible states
def states():
    for y in range(H):
        for x in range(W):
            s = State(y, x)
            if s in cliff_states:
                continue
            yield s


def target_state(s, a):
    x, y = s.x, s.y
    if a == ACTION_LEFT:
        return State(y, max(x - 1, 0))
    if a == ACTION_RIGHT:
        return State(y, min(x + 1, W - 1))
    if a == ACTION_UP:
        return State(max(y - 1, 0), x)
    if a == ACTION_DOWN:
        return State(min(y + 1, H - 1), x)


# returns a list of Transitions from the state s for each action, only non zero probabilities are given
# serves the lists for all actions at once
def transitions(s):
    if s in goal_states:
        return [[Transition(state=s, prob=1.0, reward=0)] for a in actions]

    transitions_full = []
    for a in actions:
        transitions_actions = []

        # over all *random* actions
        for a_ in actions:
            s_ = target_state(s, a_)
            if s_ in cliff_states:
                r = FALL_REWARD
                s_ = initial_state
            else:
                r = -1
            p = 1.0 - RANDOM_ACTION_P if a_ == a else RANDOM_ACTION_P / 3
            transitions_actions.append(Transition(s_, p, r))
        transitions_full.append(transitions_actions)

    return transitions_full


# gives a list of states reachable from state s by any available action
def neighbours(s):
    return {r.state for r in chain(*transitions(s).values())}



