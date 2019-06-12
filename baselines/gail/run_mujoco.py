u'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

from __future__ import division
from __future__ import absolute_import
import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier


def argsparser():
    parser = argparse.ArgumentParser(u"Tensorflow Implementation of GAIL")
    parser.add_argument(u'--env_id', help=u'environment ID', default=u'Hopper-v2')
    parser.add_argument(u'--seed', help=u'RNG seed', type=int, default=0)
    parser.add_argument(u'--expert_path', type=unicode, default=u'data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument(u'--checkpoint_dir', help=u'the directory to save model', default=u'checkpoint')
    parser.add_argument(u'--log_dir', help=u'the directory to save log file', default=u'log')
    parser.add_argument(u'--load_model_path', help=u'if provided, load the model', type=unicode, default=None)
    # Task
    parser.add_argument(u'--task', type=unicode, choices=[u'train', u'evaluate', u'sample'], default=u'train')
    # for evaluatation
    boolean_flag(parser, u'stochastic_policy', default=False, help=u'use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, u'save_sample', default=False, help=u'save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument(u'--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument(u'--g_step', help=u'number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument(u'--d_step', help=u'number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument(u'--policy_hidden_size', type=int, default=100)
    parser.add_argument(u'--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument(u'--algo', type=unicode, choices=[u'trpo', u'ppo'], default=u'trpo')
    parser.add_argument(u'--max_kl', type=float, default=0.01)
    parser.add_argument(u'--policy_entcoeff', help=u'entropy coefficiency of policy', type=float, default=0)
    parser.add_argument(u'--adversary_entcoeff', help=u'entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument(u'--save_per_iter', help=u'save model every xx iterations', type=int, default=100)
    parser.add_argument(u'--num_timesteps', help=u'number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, u'pretrained', default=False, help=u'Use BC to pretrain')
    parser.add_argument(u'--BC_max_iter', help=u'Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + u"_gail."
    if args.pretrained:
        task_name += u"with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += u"transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split(u"-")[0]
    task_name = task_name + u".g_step_" + unicode(args.g_step) + u".d_step_" + unicode(args.d_step) + \
        u".policy_entcoeff_" + unicode(args.policy_entcoeff) + u".adversary_entcoeff_" + unicode(args.adversary_entcoeff)
    task_name += u".seed_" + unicode(args.seed)
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

    if args.task == u'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    elif args.task == u'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == u'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func(u"pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(xrange(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj[u'ob'], traj[u'ac'], traj[u'ep_len'], traj[u'ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print u'stochastic policy:'
    else:
        print u'deterministic policy:'
    if save:
        filename = load_model_path.split(u'/')[-1] + u'.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print u"Average length:", avg_len
    print u"Average return:", avg_ret
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {u"ob": obs, u"rew": rews, u"new": news, u"ac": acs,
            u"ep_ret": cur_ep_ret, u"ep_len": cur_ep_len}
    return traj


if __name__ == u'__main__':
    args = argsparser()
    main(args)
