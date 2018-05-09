import argparse
import gym
import os
import numpy as np
import json

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

import cvar.dqn.core as dqn_core
from cvar.common.cvar_computation import var_cvar_from_samples
from baselines.common.misc_util import boolean_flag


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, required=True, help="load model from this directory. ")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")

    parser.add_argument("--alpha", type=str, default=1.0, help="alpha in CVaR_alpha(x_0)")
    parser.add_argument("--random-action", type=float, default=0.,
                        help="probability of selecting a random action (for more risk sensitivity)")

    parser.add_argument("--nb-episodes", type=int, default=1000, help="run how many episodes")

    return parser.parse_args()


def run(env, act, stochastic, nb_episodes):
    episode = 0
    info = {}

    obs = env.reset()

    while episode < nb_episodes:

        action = act(np.array(obs)[None], args.alpha, stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)

        if done:
            obs = env.reset()
        if len(info["rewards"]) > episode:
            episode = len(info["rewards"])
            print('{}: {}'.format(episode, info["rewards"][-1]))

    return info['rewards']


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        if args.env == 'Frogger':
            import cvar.dqn.frogger
        env, _ = dqn_core.make_env_atari(args.env)

        if args.random_action > 0:
            env = dqn_core.ActionRandomizer(env, args.random_action)

        model_parent_path = dqn_core.parent_path(args.model_dir)
        old_args = json.load(open(model_parent_path + '/args.json'))

        var_func, cvar_func = dqn_core.models.atari_model()
        act = dqn_core.build_act(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            var_func=var_func,
            cvar_func=cvar_func,
            num_actions=env.action_space.n,
            nb_atoms=old_args['nb_atoms'])
        U.load_state(os.path.join(args.model_dir, "saved"))

        rewards = run(env, act, args.stochastic, args.nb_episodes)

    print('---------------------')
    for alpha in np.arange(0.05, 1.05, 0.05):
        v, cv = var_cvar_from_samples(rewards, alpha)
        print('CVaR_{:.2f} = {}'.format(alpha, cv))
