from __future__ import division
from __future__ import absolute_import
import gym

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl[u't'] > 100 and sum(lcl[u'episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make(u"CartPole-v0")
    act = deepq.learn(
        env,
        network=u'mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print u"Saving model to cartpole_model.pkl"
    act.save(u"cartpole_model.pkl")


if __name__ == u'__main__':
    main()
