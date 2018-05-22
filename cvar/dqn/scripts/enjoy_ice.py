import gym
import numpy as np

import cvar.dqn.core as dqn_core
from cvar.dqn.core.plots import PlotMachine


def main():
    env = dqn_core.make_env_ice("IceLakeRGB-v0")
    act = dqn_core.load("../models/ice_rgb_model.pkl")

    action_set = ['Left', 'Right', 'Down', 'Up', '-']
    plot_machine = PlotMachine(act.get_nb_atoms(), env.action_space.n, action_set)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(np.array(obs)[None], 1.0, stochastic=False)[0])
            plot_machine.plot_distribution(np.array(obs)[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
