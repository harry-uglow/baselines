from __future__ import absolute_import
import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, **_3to2kwargs):
        nsteps = _3to2kwargs['nsteps']; del _3to2kwargs['nsteps']
        model = _3to2kwargs['model']; del _3to2kwargs['model']
        env = _3to2kwargs['env']; del _3to2kwargs['env']
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, u'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in xrange(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

