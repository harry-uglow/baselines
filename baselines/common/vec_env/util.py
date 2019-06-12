u"""
Helpers for dealing with vectorized environments.
"""

from __future__ import absolute_import
from collections import OrderedDict

import gym
import numpy as np


def copy_obs_dict(obs):
    u"""
    Deep-copy an observation dict.
    """
    return dict((k, np.copy(v)) for k, v in obs.items())


def dict_to_obs(obs_dict):
    u"""
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == set([None]):
        return obs_dict[None]
    return obs_dict


def obs_space_info(obs_space):
    u"""
    Get dict-structured information about a gym.Space.

    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


def obs_to_dict(obs):
    u"""
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}
