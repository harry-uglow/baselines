from __future__ import absolute_import
import pytest
import gym
import tensorflow as tf

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function
from baselines.common.tf_util import make_session

algos = [u'a2c', u'acer', u'acktr', u'deepq', u'ppo2', u'trpo_mpi']

@pytest.mark.parametrize(u'algo', algos)
def test_env_after_learn(algo):
    def make_env():
        # acktr requires too much RAM, fails on travis
        env = gym.make(u'CartPole-v1' if algo == u'acktr' else u'PongNoFrameskip-v4')
        return env

    make_session(make_default=True, graph=tf.Graph())
    env = SubprocVecEnv([make_env])

    learn = get_learn_function(algo)

    # Commenting out the following line resolves the issue, though crash happens at env.reset().
    learn(network=u'mlp', env=env, total_timesteps=0, load_path=None, seed=None)

    env.reset()
    env.close()
