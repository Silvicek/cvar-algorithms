from util.runs import epoch
from cliffwalker import *
from plots.grid import PlotMachine
from policy_improvement.policies import VarBasedQPolicy, TamarPolicy
import q_learning
from q_learning import ActionValueFunction, MarkovState

import pickle

alpha = 0.9

world, Q = pickle.load(open('../files/models/q_10_15.pkl', 'rb'))

# Q.Q[0,3,2].plot()

policy = VarBasedQPolicy(Q, alpha)
# policy = TamarPolicy(V, alpha)

# =============== plot dynamic
V_visual = q_learning.q_to_v_exp(Q)

# print(V_visual)
plot_machine = PlotMachine(world, V_visual)
# policy = var_policy
for i in range(100):
    S, A, R = epoch(world, policy, plot_machine=plot_machine)
    print('{}: {}'.format(i, np.sum(R)))
    policy.reset()