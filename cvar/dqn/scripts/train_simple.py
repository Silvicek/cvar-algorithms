import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core
import cvar.dqn.ice_lake
import argparse


def parse_args():
    parser = argparse.ArgumentParser("CVaR DQN experiments for simple environments")

    parser.add_argument("--env", type=str, default="IceLake", help="name of the game")
    parser.add_argument("--random-action", type=float, default=0., help="probability of selecting a random action (for more risk sensitivity)")
    parser.add_argument("--num-steps", type=int, default=5000, help="total number of steps to run the environment for")

    # CVaR
    parser.add_argument("--nb-atoms", type=int, default=10, help="number of cvar and quantile atoms (linearly spaced)")
    parser.add_argument("--run-alpha", type=float, default=1., help="alpha for policy used during training")

    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env+"-v0")

    if args.random_action > 0:
        env = dqn_core.ActionRandomizer(env, args.random_action)
        exploration_final_eps = 0
    else:
        exploration_final_eps = 0.02

    set_global_seeds(1337)

    var_func, cvar_func = dqn_core.models.mlp([64])
    act = dqn_core.learn(
        env,
        var_func,
        cvar_func,
        nb_atoms=5,
        run_alpha=1.0,
        lr=1e-3,
        max_timesteps=args.num_steps+1,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=exploration_final_eps,
        print_freq=10,
        batch_size=32
    )
    act.save("../models/"+args.env.lower()+"_model.pkl")

    # obs, done = env.reset(), False
    # episode_rew = 0
    # while not done:
    #     print(obs)
    #     env.render()
    #     obs, rew, done, _ = env.step(act(obs[None], 1.)[0])
    #     episode_rew += rew
    # print("Episode reward", episode_rew)
    # print('--------------')

    # ---------------------------------------------
    # import tensorflow as tf
    # tf.get_default_session().close()
    # act = dqn_core.load("../models/" + args.env.lower() + "_model.pkl")
    # env = gym.make(args.env + "-v0")

    # while True:
    #     obs, done = env.reset(), False
    #     episode_rew = 0
    #     while not done:
    #         env.render()
    #         a = act(obs[None], args.run_alpha)[0]
    #         print(obs, '->', a)
    #         obs, rew, done, _ = env.step(a)
    #         episode_rew += rew
    #     print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
