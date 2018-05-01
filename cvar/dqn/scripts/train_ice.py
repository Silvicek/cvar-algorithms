import cvar.dqn.core as dqn_core
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import FrameStack, WarpFrame
import gym
import cvar.dqn.ice_lake


def make_env(game_name):
    env = gym.make(game_name)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    return env


def main():
    set_global_seeds(1337)
    env = make_env("IceLakeRGB-v0")

    var_func, cvar_func = dqn_core.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
    )
    act = dqn_core.learn(
        env,
        run_alpha=1.0,
        var_func=var_func,
        cvar_func=cvar_func,
        lr=1e-4,
        max_timesteps=int(2e6),
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        batch_size=32,
        nb_atoms=50,
        print_freq=25
    )
    act.save("../models/ice_rgb_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
