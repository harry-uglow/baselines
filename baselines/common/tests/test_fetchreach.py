from __future__ import absolute_import
import pytest
import gym

from baselines.run import get_learn_function
from baselines.common.tests.util import reward_per_episode_test

pytest.importorskip(u'mujoco_py')

common_kwargs = dict(
    network=u'mlp',
    seed=0,
)

learn_kwargs = {
    u'her': dict(total_timesteps=2000)
}

@pytest.mark.slow
@pytest.mark.parametrize(u"alg", learn_kwargs.keys())
def test_fetchreach(alg):
    u'''
    Test if the algorithm (with an mlp policy)
    can learn the FetchReach task
    '''

    kwargs = common_kwargs.copy()
    kwargs.update(learn_kwargs[alg])

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    def env_fn():

        env = gym.make(u'FetchReach-v1')
        env.seed(0)
        return env

    reward_per_episode_test(env_fn, learn_fn, -15)

if __name__ == u'__main__':
    test_fetchreach(u'her')
