import gym
from gym.envs.registration import register
import cvar.dqn.core as dqn_core
from cvar.dqn.core.static import ActionRandomizer


def main():
    register(
        id='DeterministicMountainCar-v0',
        entry_point='cvar.dqn.train_mountaincar:DeterministicMountainCarEnv',
        max_episode_steps=200,
        reward_threshold=-110.0,
    )
    env = gym.make("DeterministicMountainCar-v0")
    env = ActionRandomizer(env, eps=0.1)
    act = dqn_core.load("models/mountaincar_model.pkl")
    action_set = dqn_core.actions_from_env(env)
    plot_machine = dqn_core.PlotMachine(act.get_nb_atoms(), env.action_space.n, action_set)
    alpha = 1.0
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None], alpha)[0])
            plot_machine.plot_distribution(obs[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
