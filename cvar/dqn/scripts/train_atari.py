import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import json

import baselines.common.tf_util as U

from baselines import logger
import cvar.dqn.core as dqn_core
from cvar.dqn.core.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from cvar.dqn.core.simple import make_session
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    parser.add_argument("--random-action", type=float, default=0., help="probability of selecting a random action (for more risk sensitivity)")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=10000, help="number of iterations between every target network update")
    # Bells and whistles
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False, help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    # CVaR
    parser.add_argument("--nb-atoms", type=int, default=10, help="number of cvar and quantile atoms (linearly spaced)")
    parser.add_argument("--run-alpha", type=float, default=1., help="alpha for policy used during training")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved.")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def maybe_save_model(savedir, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))

    # requires 32+gb of memory
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()

    # Parse savedir
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)

    # Create and seed the env.
    env, monitored_env = dqn_core.make_env_atari(args.env)
    if args.random_action > 0:
        env = dqn_core.ActionRandomizer(env, args.random_action)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    if args.gym_monitor and savedir:
        env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    var_func, cvar_func = dqn_core.models.atari_model()

    sess = make_session(num_cpu=4)
    sess.__enter__()

    # Create training graph
    act, train, update_target, debug = dqn_core.build_train(
        make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
        var_func=var_func,
        cvar_func=cvar_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
        gamma=0.99,
        nb_atoms=args.nb_atoms
    )

    # Create the schedule for exploration starting from 1.
    final_p = 0 if args.random_action > 0 else 0.01
    exploration = LinearSchedule(schedule_timesteps=int(0.1 * args.num_steps),
                                 initial_p=1.0,
                                 final_p=final_p)
    # approximate_num_iters = args.num_steps / 4
    # exploration = PiecewiseSchedule([
    #     (0, 1.0),
    #     (approximate_num_iters / 50, 0.1),
    #     (approximate_num_iters / 5, 0.01)
    # ], outside_value=0.01)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    U.initialize()
    update_target()
    num_iters = 0

    # Load the model
    state = maybe_load_model(savedir)
    if state is not None:
        num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
        monitored_env.set_state(state["monitor_state"])

    start_time, start_steps = None, None
    steps_per_iter = RunningAvg(0.999)
    iteration_time_est = RunningAvg(0.999)
    obs = env.reset()
    num_iters_since_reset = 0
    reset = True

    # Main training loop
    while True:
        num_iters += 1
        num_iters_since_reset += 1

        # Take action and store transition in the replay buffer.
        kwargs = {}

        update_eps = exploration.value(num_iters)
        update_param_noise_threshold = 0.

        action = act(np.array(obs)[None], args.run_alpha, update_eps=update_eps, **kwargs)[0]
        reset = False
        new_obs, rew, done, info = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        if done:
            num_iters_since_reset = 0
            obs = env.reset()
            reset = True

        if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                num_iters % args.learning_freq == 0):
            # Sample a bunch of transitions from replay buffer
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
            weights = np.ones_like(rewards)
            # Minimize the error in Bellman's equation and compute TD-error
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

        # Update target network.
        if num_iters % args.target_update_freq == 0:
            update_target()

        if start_time is not None:
            steps_per_iter.update(info['steps'] - start_steps)
            iteration_time_est.update(time.time() - start_time)
        start_time, start_steps = time.time(), info["steps"]

        # Save the model and training state.
        if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
            maybe_save_model(savedir, {
                'replay_buffer': replay_buffer,
                'num_iters': num_iters,
                'monitor_state': monitored_env.get_state(),
            })

        if info["steps"] > args.num_steps:
            break

        if done:
            steps_left = args.num_steps - info["steps"]
            completion = np.round(100*info["steps"] / args.num_steps, 2)

            logger.record_tabular("% completion", completion)
            logger.record_tabular("steps", info["steps"])
            logger.record_tabular("iters", num_iters)
            logger.record_tabular("episodes", len(info["rewards"]))
            logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
            logger.record_tabular("exploration", exploration.value(num_iters))

            fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                            if steps_per_iter._value is not None else "calculating...")
            logger.dump_tabular()
            logger.log()
            logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
            logger.log()
