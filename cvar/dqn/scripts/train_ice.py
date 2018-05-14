import cvar.dqn.core as dqn_core
from baselines.common import set_global_seeds
from cvar.dqn.core.static import make_env_ice


def main():
    # set_global_seeds(1337)
    env = make_env_ice("IceLakeRGB-v0")

    var_func, cvar_func = dqn_core.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
    )
    act = dqn_core.learn(
        env,
        var_func=var_func,
        cvar_func=cvar_func,
        lr=1e-5,
        max_timesteps=10000000 + 1,
        buffer_size=500000,
        exploration_fraction=0.2,
        exploration_final_eps=0.3,
        train_freq=4,
        learning_starts=100000,
        target_network_update_freq=1000,
        gamma=0.99,
        batch_size=32,
        nb_atoms=100,
        print_freq=25,
        periodic_save_path="../models/ice_rgb",
        periodic_save_freq=500000,
        grad_norm_clip=10.
    )
    act.save("../models/ice_rgb_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
