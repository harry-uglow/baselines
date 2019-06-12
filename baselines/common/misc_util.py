from __future__ import with_statement
from __future__ import absolute_import
import gym
import numpy as np
import os
import pickle
import random
import tempfile
import zipfile
from itertools import izip
from io import open


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return izip(*seqs)


class EzPickle(object):
    u"""Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {u"_ezpickle_args": self._ezpickle_args, u"_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d[u"_ezpickle_args"], **d[u"_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def pretty_eta(seconds_left):
    u"""Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return u"{} {}{}".format(unicode(cnt), name, (u's' if cnt > 1 else u''))

    if days_left > 0:
        msg = helper(days_left, u'day')
        if hours_left > 0:
            msg += u' and ' + helper(hours_left, u'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, u'hour')
        if minutes_left > 0:
            msg += u' and ' + helper(minutes_left, u'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, u'minute')
    return u'less than a minute'


class RunningAvg(object):
    def __init__(self, gamma, init_value=None):
        u"""Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.

        Parameters
        ----------
        gamma: float
            Must be between 0 and 1, where 0 is the most sensitive to recent
            changes.
        init_value: float or None
            Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        u"""Update the estimate.

        Parameters
        ----------
        new_val: float
            new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        u"""Get the current estimate"""
        return self._value

def boolean_flag(parser, name, default=False, help=None):
    u"""Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace(u'-', u'_')
    parser.add_argument(u"--" + name, action=u"store_true", default=default, dest=dest, help=help)
    parser.add_argument(u"--no-" + name, action=u"store_false", dest=dest)


def get_wrapper_by_name(env, classname):
    u"""Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    Parameters
    ----------
    env: gym.Env of gym.Wrapper
        gym environment
    classname: str
        name of the wrapper

    Returns
    -------
    wrapper: gym.Wrapper
        wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError(u"Couldn't find wrapper named %s" % classname)


def relatively_safe_pickle_dump(obj, path, compression=False):
    u"""This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The indended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    Parameters
    ----------
    obj: object
        object to pickle
    path: str
        path to the output file
    compression: bool
        if true pickle will be compressed
    """
    temp_storage = path + u".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            uncompressed_file.file.flush()
            with zipfile.ZipFile(temp_storage, u"w", compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, u"data")
    else:
        with open(temp_storage, u"wb") as f:
            pickle.dump(obj, f)
    os.rename(temp_storage, path)


def pickle_load(path, compression=False):
    u"""Unpickle a possible compressed pickle.

    Parameters
    ----------
    path: str
        path to the output file
    compression: bool
        if true assumes that pickle was compressed when created and attempts decompression.

    Returns
    -------
    obj: object
        the unpickled object
    """

    if compression:
        with zipfile.ZipFile(path, u"r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open(u"data") as f:
                return pickle.load(f)
    else:
        with open(path, u"rb") as f:
            return pickle.load(f)
