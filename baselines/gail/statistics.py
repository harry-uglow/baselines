u'''
This code is highly based on https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/statistic.py
'''

from __future__ import with_statement
from __future__ import absolute_import
import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from itertools import izip


class stats(object):

    def __init__(self, scalar_keys=[], histogram_keys=[]):
        self.scalar_keys = scalar_keys
        self.histogram_keys = histogram_keys
        self.scalar_summaries = []
        self.scalar_summaries_ph = []
        self.histogram_summaries_ph = []
        self.histogram_summaries = []
        with tf.variable_scope(u'summary'):
            for k in scalar_keys:
                ph = tf.placeholder(u'float32', None, name=k+u'.scalar.summary')
                sm = tf.summary.scalar(k+u'.scalar.summary', ph)
                self.scalar_summaries_ph.append(ph)
                self.scalar_summaries.append(sm)
            for k in histogram_keys:
                ph = tf.placeholder(u'float32', None, name=k+u'.histogram.summary')
                sm = tf.summary.scalar(k+u'.histogram.summary', ph)
                self.histogram_summaries_ph.append(ph)
                self.histogram_summaries.append(sm)

        self.summaries = tf.summary.merge(self.scalar_summaries+self.histogram_summaries)

    def add_all_summary(self, writer, values, iter):
        # Note that the order of the incoming ```values``` should be the same as the that of the
        #            ```scalar_keys``` given in ```__init__```
        if np.sum(np.isnan(values)+0) != 0:
            return
        sess = U.get_session()
        keys = self.scalar_summaries_ph + self.histogram_summaries_ph
        feed_dict = {}
        for k, v in izip(keys, values):
            feed_dict.update({k: v})
        summaries_str = sess.run(self.summaries, feed_dict)
        writer.add_summary(summaries_str, iter)
