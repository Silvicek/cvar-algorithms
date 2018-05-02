import cvar.dqn.core as dqn_core
import numpy as np
from cvar.common.plots import PlotMachine
from cvar.dqn.scripts.train_ice import make_env

def main():
    env = make_env("IceLakeRGB-v0")
    act = dqn_core.load("../models/ice_rgb_model.pkl")

    action_set = dqn_core.actions_from_env(env)
    plot_machine = PlotMachine(act.get_nb_atoms(), env.action_space.n, action_set)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(np.array(obs)[None])[0])
            plot_machine.plot_distribution(np.array(obs)[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
