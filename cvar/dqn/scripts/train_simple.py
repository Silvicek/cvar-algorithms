import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core
import argparse
import cvar.dqn.ice_lake


def parse_args():
    parser = argparse.ArgumentParser("CVaR DQN experiments for simple environments")

    parser.add_argument("--env", type=str, default="IceLake", help="name of the game")
    parser.add_argument("--random-action", type=float, default=0., help="probability of selecting a random action (for more risk sensitivity)")
    parser.add_argument("--num-steps", type=int, default=50000, help="total number of steps to run the environment for")
    parser.add_argument("--buffer-size", type=int, default=50000, help="size of replay memory")

    # CVaR
    parser.add_argument("--nb-atoms", type=int, default=10, help="number of cvar and quantile atoms (linearly spaced)")
    parser.add_argument("--run-alpha", type=float, default=1., help="alpha for policy used during training. -1 "
                                                                    "means")

    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env+"-v0")

    if args.random_action > 0:
        env = dqn_core.ActionRandomizer(env, args.random_action)
        exploration_final_eps = 0
    else:
        exploration_final_eps = 0.3

    set_global_seeds(1337)

    var_func, cvar_func = dqn_core.models.mlp([64])
    act = dqn_core.learn(
        env,
        var_func,
        cvar_func,
        nb_atoms=args.nb_atoms,
        run_alpha=args.run_alpha if args.run_alpha > 0 else None,
        lr=1e-4,
        max_timesteps=args.num_steps+1,
        buffer_size=args.buffer_size,
        exploration_fraction=0.2,
        exploration_final_eps=exploration_final_eps,
        print_freq=10,
        batch_size=32,
        periodic_save_path="../models/"+args.env.lower()
    )
    act.save("../models/"+args.env.lower()+"_model.pkl")


if __name__ == '__main__':
    main()
