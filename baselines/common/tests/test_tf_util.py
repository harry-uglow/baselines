# tests for tf_util
from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
from baselines.common.tf_util import (
    function,
    initialize,
    single_threaded_session
)


def test_function():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.int32, (), name=u"x")
        y = tf.placeholder(tf.int32, (), name=u"y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12


def test_multikwargs():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.int32, (), name=u"x")
        with tf.variable_scope(u"other"):
            x2 = tf.placeholder(tf.int32, (), name=u"x")
        z = 3 * x + 2 * x2

        lin = function([x, x2], z, givens={x2: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(2, 2) == 10


if __name__ == u'__main__':
    test_function()
    test_multikwargs()
