from cvar.gridworld.cliffwalker import *
from cvar.gridworld.plots.grid import InteractivePlotMachine
from cvar.gridworld.core.policies import VarBasedQPolicy, XiBasedPolicy
from cvar.gridworld.algorithms.q_learning import q_learning

if __name__ == '__main__':
    import pickle
    np.random.seed(2)

    # ============================= new config
    run_alpha = 1.
    world = GridWorld(10, 15, random_action_p=0.1)
    np.random.seed(20)
    Q = q_learning(world, run_alpha, max_episodes=10000)

    pickle.dump((world, Q), open('data/models/q_10_15.pkl', mode='wb'))

    # ============================= load
    world, Q = pickle.load(open('data/models/q_10_15.pkl', 'rb'))

    # ============================= RUN
    print('ATOMS:', Q.atoms)

    for alpha in np.arange(0.05, 1.05, 0.05):
        print(alpha)
        pm = InteractivePlotMachine(world, Q, alpha=alpha, action_value=True)
        pm.show()

    # =============== plot dynamic
    # V_visual = q_learning.q_to_v_exp(Q)
    #
    # # print(V_visual)
    # plot_machine = PlotMachine(world, V_visual)
    # # policy = var_policy
    # for i in range(100):
    #     S, A, R = epoch(world, policy, plot_machine=plot_machine)
    #     print('{}: {}'.format(i, np.sum(R)))
    #     policy.reset()