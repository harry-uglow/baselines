from __future__ import with_statement
from __future__ import absolute_import
import os
import gym
import tempfile
import pytest
import tensorflow as tf
import numpy as np

from baselines.common.tests.envs.mnist_env import MnistEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.run import get_learn_function
from baselines.common.tf_util import make_session, get_session

from functools import partial
from itertools import izip


learn_kwargs = {
    u'deepq': {},
    u'a2c': {},
    u'acktr': {},
    u'acer': {},
    u'ppo2': {u'nminibatches': 1, u'nsteps': 10},
    u'trpo_mpi': {},
}

network_kwargs = {
    u'mlp': {},
    u'cnn': {u'pad': u'SAME'},
    u'lstm': {},
    u'cnn_lnlstm': {u'pad': u'SAME'}
}


@pytest.mark.parametrize(u"learn_fn", learn_kwargs.keys())
@pytest.mark.parametrize(u"network_fn", network_kwargs.keys())
def test_serialization(learn_fn, network_fn):
    u'''
    Test if the trained model can be serialized
    '''


    if network_fn.endswith(u'lstm') and learn_fn in [u'acer', u'acktr', u'trpo_mpi', u'deepq']:
            # TODO make acktr work with recurrent policies
            # and test
            # github issue: https://github.com/openai/baselines/issues/660
            return

    def make_env():
        env = MnistEnv(episode_len=100)
        env.seed(10)
        return env

    env = DummyVecEnv([make_env])
    ob = env.reset().copy()
    learn = get_learn_function(learn_fn)

    kwargs = {}
    kwargs.update(network_kwargs[network_fn])
    kwargs.update(learn_kwargs[learn_fn])


    learn = partial(learn, env=env, network=network_fn, seed=0, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, u'serialization_test_model')

        with tf.Graph().as_default(), make_session().as_default():
            model = learn(total_timesteps=100)
            model.save(model_path)
            mean1, std1 = _get_action_stats(model, ob)
            variables_dict1 = _serialize_variables()

        with tf.Graph().as_default(), make_session().as_default():
            model = learn(total_timesteps=0, load_path=model_path)
            mean2, std2 = _get_action_stats(model, ob)
            variables_dict2 = _serialize_variables()

        for k, v in variables_dict1.items():
            np.testing.assert_allclose(v, variables_dict2[k], atol=0.01,
                err_msg=u'saved and loaded variable {} value mismatch'.format(k))

        np.testing.assert_allclose(mean1, mean2, atol=0.5)
        np.testing.assert_allclose(std1, std2, atol=0.5)


@pytest.mark.parametrize(u"learn_fn", learn_kwargs.keys())
@pytest.mark.parametrize(u"network_fn", [u'mlp'])
def test_coexistence(learn_fn, network_fn):
    u'''
    Test if more than one model can exist at a time
    '''

    if learn_fn == u'deepq':
            # TODO enable multiple DQN models to be useable at the same time
            # github issue https://github.com/openai/baselines/issues/656
            return

    if network_fn.endswith(u'lstm') and learn_fn in [u'acktr', u'trpo_mpi', u'deepq']:
            # TODO make acktr work with recurrent policies
            # and test
            # github issue: https://github.com/openai/baselines/issues/660
            return

    env = DummyVecEnv([lambda: gym.make(u'CartPole-v0')])
    learn = get_learn_function(learn_fn)

    kwargs = {}
    kwargs.update(network_kwargs[network_fn])
    kwargs.update(learn_kwargs[learn_fn])

    learn =  partial(learn, env=env, network=network_fn, total_timesteps=0, **kwargs)
    make_session(make_default=True, graph=tf.Graph())
    model1 = learn(seed=1)
    make_session(make_default=True, graph=tf.Graph())
    model2 = learn(seed=2)

    model1.step(env.observation_space.sample())
    model2.step(env.observation_space.sample())



def _serialize_variables():
    sess = get_session()
    variables = tf.trainable_variables()
    values = sess.run(variables)
    return dict((var.name, value) for var, value in izip(variables, values))


def _get_action_stats(model, ob):
    ntrials = 1000
    if model.initial_state is None or model.initial_state == []:
        actions = np.array([model.step(ob)[0] for _ in xrange(ntrials)])
    else:
        actions = np.array([model.step(ob, S=model.initial_state, M=[False])[0] for _ in xrange(ntrials)])

    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0)

    return mean, std

