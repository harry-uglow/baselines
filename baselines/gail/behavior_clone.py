u'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

from __future__ import division
from __future__ import absolute_import
import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.run_mujoco import runner
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset


def argsparser():
    parser = argparse.ArgumentParser(u"Tensorflow Implementation of Behavior Cloning")
    parser.add_argument(u'--env_id', help=u'environment ID', default=u'Hopper-v1')
    parser.add_argument(u'--seed', help=u'RNG seed', type=int, default=0)
    parser.add_argument(u'--expert_path', type=unicode, default=u'data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument(u'--checkpoint_dir', help=u'the directory to save model', default=u'checkpoint')
    parser.add_argument(u'--log_dir', help=u'the directory to save log file', default=u'log')
    #  Mujoco Dataset Configuration
    parser.add_argument(u'--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument(u'--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, u'stochastic_policy', default=False, help=u'use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, u'save_sample', default=False, help=u'save the trajectories or not')
    parser.add_argument(u'--BC_max_iter', help=u'Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func(u"pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name=u"ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name=u"stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log(u"Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(xrange(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, u'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, u'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log(u"Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_state(savedir_fname, var_list=pi.get_variables())
    return savedir_fname


def get_task_name(args):
    task_name = u'BC'
    task_name += u'.{}'.format(args.env_id.split(u"-")[0])
    task_name += u'.traj_limitation_{}'.format(args.traj_limitation)
    task_name += u".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), u"monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname = learn(env,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    avg_len, avg_ret = runner(env,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=1024,
                              number_trajs=10,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)


if __name__ == u'__main__':
    args = argsparser()
    main(args)
