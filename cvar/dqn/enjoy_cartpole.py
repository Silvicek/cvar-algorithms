import gym

import cvar.dqn.core as dqn_core


def main():
    env = gym.make("CartPole-v0")
    act = dqn_core.load("cartpole_model.pkl")
    action_set = dqn_core.actions_from_env(env)
    plot_machine = dqn_core.PlotMachine(act.get_dist_params(), env.action_space.n, action_set)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            plot_machine.plot_distribution(obs[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
