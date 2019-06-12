from __future__ import absolute_import
import gym

from baselines import deepq
from baselines.common import models


def main():
    env = gym.make(u"MountainCar-v0")
    act = deepq.learn(
        env,
        network=models.mlp(num_layers=1, num_hidden=64),
        total_timesteps=0,
        load_path=u'mountaincar_model.pkl'
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print u"Episode reward", episode_rew


if __name__ == u'__main__':
    main()
