import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("MountainCar-v0")
    set_global_seeds(1337)

    var_func, cvar_func = dqn_core.models.mlp([64])
    act = dqn_core.learn(
        env,
        var_func,
        cvar_func,
        run_alpha=1.0,
        lr=1e-3,
        max_timesteps=400000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        print_freq=10,
        callback=callback,
        batch_size=32,
        dist_params={'nb_atoms': 5, 'huber_loss': False}
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("models/mountaincar_model.pkl")


if __name__ == '__main__':
    main()
