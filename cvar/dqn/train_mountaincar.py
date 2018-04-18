import gym
from baselines.common import set_global_seeds
import cvar.dqn.core as dqn_core
import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.registration import register
from cvar.dqn.core.static import ActionRandomizer


class DeterministicMountainCarEnv(MountainCarEnv):
    def _reset(self):
        self.state = np.array([-0.3, 0])
        return np.array(self.state)


def main():
    set_global_seeds(1337)
    register(
        id='DeterministicMountainCar-v0',
        entry_point='cvar.dqn.train_mountaincar:DeterministicMountainCarEnv',
        max_episode_steps=200,
        reward_threshold=-110.0,
    )
    # env = gym.make("MountainCar-v0")
    env = gym.make("DeterministicMountainCar-v0")
    env = ActionRandomizer(env, eps=0.1)
    # env = DeterministicMountainCarEnv()

    random_action_eps = 0.
    if random_action_eps > 0:
        env = dqn_core.ActionRandomizer(env, random_action_eps)


    var_func, cvar_func = dqn_core.models.mlp([64])
    act = dqn_core.learn(
        env,
        var_func,
        cvar_func,
        nb_atoms=1,
        run_alpha=1.0,
        lr=1e-3,
        max_timesteps=300000,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.,
        print_freq=10,
        batch_size=32
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("models/mountaincar_model.pkl")


if __name__ == '__main__':
    main()
