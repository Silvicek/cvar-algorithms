import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core


def main():
    env = gym.make("MountainCar-v0")
    random_action_eps = 0.
    if random_action_eps > 0:
        env = dqn_core.ActionRandomizer(env, random_action_eps)
    set_global_seeds(1337)

    var_func, cvar_func = dqn_core.models.mlp([64])
    act = dqn_core.learn(
        env,
        var_func,
        cvar_func,
        nb_atoms=5,
        run_alpha=1.0,
        lr=1e-3,
        max_timesteps=300000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        print_freq=10,
        batch_size=32
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("models/mountaincar_model.pkl")


if __name__ == '__main__':
    main()
