import argparse
import gym
import os
import numpy as np
import json

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

import cvar.dqn.core as dqn_core
from cvar.common.plots import PlotMachine
from baselines.common.misc_util import boolean_flag


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "visual", default=False, help="whether or not to show the distribution output")

    parser.add_argument("--alpha", type=str, default=1.0, help="alpha in CVaR_alpha(x_0)")

    return parser.parse_args()


def play(env, act, stochastic, video_path, nb_atoms):
    num_episodes = 0
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    if args.visual:
        action_names = dqn_core.actions_from_env(env)
        plot_machine = PlotMachine(nb_atoms, env.action_space.n, action_names)
    while True:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], args.alpha, stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)
        if args.visual:
            plot_machine.plot_distribution(np.array(obs)[None])

        if done:
            obs = env.reset()
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])
        # input()


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env, _ = dqn_core.make_env(args.env)

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
        play(env, act, args.stochastic, args.video, old_args['nb_atoms'])
