from util.runs import epoch
from cliffwalker import *
from plots.grid_plot_machine import PlotMachine
from policy_improvement.policies import VarBasedQPolicy
import q_learning

import pickle


if __name__ == '__main__':

    world = GridWorld(4, 6, random_action_p=0.1)
    alpha = 0.1
    # =============== VI setup
    # Q = q_learning.q_learning(world, alpha, max_episodes=4e3)
    # pickle.dump(Q, open("../files/q.pkl", 'wb'))

    Q = pickle.load(open("../files/q.pkl", 'rb'))

    Q.Q[0,3,2].plot()

    policy = VarBasedQPolicy(Q, alpha)

    # =============== plot dynamic
    V_visual = q_learning.q_to_v_exp(Q)

    # print(V_visual)
    plot_machine = PlotMachine(world, V_visual)
    # policy = var_policy
    for i in range(100):
        S, A, R = epoch(world, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()