import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core
import gym_tetris
import atari_py
from gym.envs.registration import register

register(
    id='FroggerNoFrameskip-v4',
    entry_point='gym.envs.atari:AtariEnv',
    kwargs={'game': 'frogger', 'obs_type': 'image', 'frameskip': 1},
    max_episode_steps=100000,
    nondeterministic=False,
)


def main():
    # env, _ = dqn_core.make_env("Frogger", clip_rewards=False)
    env = gym.make("FroggerNoFrameskip-v4")

    random_action_eps = 0.
    if random_action_eps > 0:
        env = dqn_core.ActionRandomizer(env, random_action_eps)

    var_func, cvar_func = dqn_core.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        # hiddens=[512],
    )
    act = dqn_core.learn(
        env,
        run_alpha=1.0,
        var_func=var_func,
        cvar_func=cvar_func,
        lr=1e-4,
        max_timesteps=int(2e6),
        # max_timesteps=100,
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        batch_size=32,
        nb_atoms=50
    )
    act.save("models/frogger_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
