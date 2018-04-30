import gym
from gym.envs.registration import register
import cvar.dqn.core as dqn_core
from cvar.dqn.core.static import ActionRandomizer
from cvar.common.plots import PlotMachine
import argparse
from baselines.common.misc_util import boolean_flag
import cvar.dqn.ice_lake


def parse_args():
    parser = argparse.ArgumentParser("CVaR DQN experiments for simple environments")

    parser.add_argument("--env", type=str, default="IceLake", help="name of the game")
    parser.add_argument("--random-action", type=float, default=0., help="probability of selecting a random action (for more risk sensitivity)")
    parser.add_argument("--num-steps", type=int, default=50000, help="total number of steps to run the environment for")

    boolean_flag(parser, "visual", default=True, help="whether or not to show the distribution plots")

    # CVaR
    parser.add_argument("--nb-atoms", type=int, default=10, help="number of cvar and quantile atoms (linearly spaced)")
    parser.add_argument("--run-alpha", type=float, default=1., help="alpha for policy used during training")

    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env+"-v0")
    if args.random_action > 0:
        env = dqn_core.ActionRandomizer(env, args.random_action)

    act = dqn_core.load("../models/"+args.env.lower()+"_model.pkl")
    action_set = dqn_core.actions_from_env(env)
    plot_machine = PlotMachine(act.get_nb_atoms(), env.action_space.n, action_set)
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None], args.run_alpha)[0])
            if args.visual:
                plot_machine.plot_distribution(obs[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
