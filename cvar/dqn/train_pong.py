import cvar.dqn.core as dqn_core
from baselines.common import set_global_seeds


def main():
    set_global_seeds(1337)
    env, _ = dqn_core.make_env("Pong")

    model = dqn_core.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False
    )
    act = dqn_core.learn(
        env,
        quant_func=model,
        lr=1e-4,
        max_timesteps=int(2e6),
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        batch_size=32,
        dist_params={'nb_atoms': 10, 'huber_loss': True}
    )
    act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
