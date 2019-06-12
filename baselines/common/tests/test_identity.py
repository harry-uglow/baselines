from __future__ import absolute_import
import pytest
from baselines.common.tests.envs.identity_env import DiscreteIdentityEnv, BoxIdentityEnv, MultiDiscreteIdentityEnv
from baselines.run import get_learn_function
from baselines.common.tests.util import simple_test

common_kwargs = dict(
    total_timesteps=30000,
    network=u'mlp',
    gamma=0.9,
    seed=0,
)

learn_kwargs = {
    u'a2c' : {},
    u'acktr': {},
    u'deepq': {},
    u'ddpg': dict(layer_norm=True),
    u'ppo2': dict(lr=1e-3, nsteps=64, ent_coef=0.0),
    u'trpo_mpi': dict(timesteps_per_batch=100, cg_iters=10, gamma=0.9, lam=1.0, max_kl=0.01)
}


algos_disc = [u'a2c', u'acktr', u'deepq', u'ppo2', u'trpo_mpi']
algos_multidisc = [u'a2c', u'acktr', u'ppo2', u'trpo_mpi']
algos_cont = [u'a2c', u'acktr', u'ddpg',  u'ppo2', u'trpo_mpi']

@pytest.mark.slow
@pytest.mark.parametrize(u"alg", algos_disc)
def test_discrete_identity(alg):
    u'''
    Test if the algorithm (with an mlp policy)
    can learn an identity transformation (i.e. return observation as an action)
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    env_fn = lambda: DiscreteIdentityEnv(10, episode_len=100)
    simple_test(env_fn, learn_fn, 0.9)

@pytest.mark.slow
@pytest.mark.parametrize(u"alg", algos_multidisc)
def test_multidiscrete_identity(alg):
    u'''
    Test if the algorithm (with an mlp policy)
    can learn an identity transformation (i.e. return observation as an action)
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    env_fn = lambda: MultiDiscreteIdentityEnv((3,3), episode_len=100)
    simple_test(env_fn, learn_fn, 0.9)

@pytest.mark.slow
@pytest.mark.parametrize(u"alg", algos_cont)
def test_continuous_identity(alg):
    u'''
    Test if the algorithm (with an mlp policy)
    can learn an identity transformation (i.e. return observation as an action)
    to a required precision
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)
    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)

    env_fn = lambda: BoxIdentityEnv((1,), episode_len=100)
    simple_test(env_fn, learn_fn, -0.1)

if __name__ == u'__main__':
    test_multidiscrete_identity(u'acktr')

