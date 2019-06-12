from __future__ import absolute_import
import re
import os.path as osp
import os
from itertools import ifilter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_atari7 = [u'BeamRider', u'Breakout', u'Enduro', u'Pong', u'Qbert', u'Seaquest', u'SpaceInvaders']
_atariexpl7 = [u'Freeway', u'Gravitar', u'MontezumaRevenge', u'Pitfall', u'PrivateEye', u'Solaris', u'Venture']

_BENCHMARKS = []

remove_version_re = re.compile(ur'-v\d+$')


def register_benchmark(benchmark):
    for b in _BENCHMARKS:
        if b[u'name'] == benchmark[u'name']:
            raise ValueError(u'Benchmark with name %s already registered!' % b[u'name'])

    # automatically add a description if it is not present
    if u'tasks' in benchmark:
        for t in benchmark[u'tasks']:
            if u'desc' not in t:
                t[u'desc'] = remove_version_re.sub(u'', t.get(u'env_id', t.get(u'id')))
    _BENCHMARKS.append(benchmark)


def list_benchmarks():
    return [b[u'name'] for b in _BENCHMARKS]


def get_benchmark(benchmark_name):
    for b in _BENCHMARKS:
        if b[u'name'] == benchmark_name:
            return b
    raise ValueError(u'%s not found! Known benchmarks: %s' % (benchmark_name, list_benchmarks()))


def get_task(benchmark, env_id):
    u"""Get a task by env_id. Return None if the benchmark doesn't have the env"""
    return ifilter(lambda task: task[u'env_id'] == env_id, benchmark[u'tasks']), None.next()


def find_task_for_env_id_in_any_benchmark(env_id):
    for bm in _BENCHMARKS:
        for task in bm[u"tasks"]:
            if task[u"env_id"] == env_id:
                return bm, task
    return None, None


_ATARI_SUFFIX = u'NoFrameskip-v4'

register_benchmark({
    u'name': u'Atari50M',
    u'description': u'7 Atari games from Mnih et al. (2013), with pixel observations, 50M timesteps',
    u'tasks': [{u'desc': _game, u'env_id': _game + _ATARI_SUFFIX, u'trials': 2, u'num_timesteps': int(50e6)} for _game in _atari7]
})

register_benchmark({
    u'name': u'Atari10M',
    u'description': u'7 Atari games from Mnih et al. (2013), with pixel observations, 10M timesteps',
    u'tasks': [{u'desc': _game, u'env_id': _game + _ATARI_SUFFIX, u'trials': 6, u'num_timesteps': int(10e6)} for _game in _atari7]
})

register_benchmark({
    u'name': u'Atari1Hr',
    u'description': u'7 Atari games from Mnih et al. (2013), with pixel observations, 1 hour of walltime',
    u'tasks': [{u'desc': _game, u'env_id': _game + _ATARI_SUFFIX, u'trials': 2, u'num_seconds': 60 * 60} for _game in _atari7]
})

register_benchmark({
    u'name': u'AtariExploration10M',
    u'description': u'7 Atari games emphasizing exploration, with pixel observations, 10M timesteps',
    u'tasks': [{u'desc': _game, u'env_id': _game + _ATARI_SUFFIX, u'trials': 2, u'num_timesteps': int(10e6)} for _game in _atariexpl7]
})


# MuJoCo

_mujocosmall = [
    u'InvertedDoublePendulum-v2', u'InvertedPendulum-v2',
    u'HalfCheetah-v2', u'Hopper-v2', u'Walker2d-v2',
    u'Reacher-v2', u'Swimmer-v2']
register_benchmark({
    u'name': u'Mujoco1M',
    u'description': u'Some small 2D MuJoCo tasks, run for 1M timesteps',
    u'tasks': [{u'env_id': _envid, u'trials': 6, u'num_timesteps': int(1e6)} for _envid in _mujocosmall]
})

register_benchmark({
    u'name': u'MujocoWalkers',
    u'description': u'MuJoCo forward walkers, run for 8M, humanoid 100M',
    u'tasks': [
        {u'env_id': u"Hopper-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
        {u'env_id': u"Walker2d-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
        {u'env_id': u"Humanoid-v1", u'trials': 4, u'num_timesteps': 100 * 1000000},
    ]
})

# Bullet
_bulletsmall = [
    u'InvertedDoublePendulum', u'InvertedPendulum', u'HalfCheetah', u'Reacher', u'Walker2D', u'Hopper', u'Ant'
]
_bulletsmall = [e + u'BulletEnv-v0' for e in _bulletsmall]

register_benchmark({
    u'name': u'Bullet1M',
    u'description': u'6 mujoco-like tasks from bullet, 1M steps',
    u'tasks': [{u'env_id': e, u'trials': 6, u'num_timesteps': int(1e6)} for e in _bulletsmall]
})


# Roboschool

register_benchmark({
    u'name': u'Roboschool8M',
    u'description': u'Small 2D tasks, up to 30 minutes to complete on 8 cores',
    u'tasks': [
        {u'env_id': u"RoboschoolReacher-v1", u'trials': 4, u'num_timesteps': 2 * 1000000},
        {u'env_id': u"RoboschoolAnt-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
        {u'env_id': u"RoboschoolHalfCheetah-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
        {u'env_id': u"RoboschoolHopper-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
        {u'env_id': u"RoboschoolWalker2d-v1", u'trials': 4, u'num_timesteps': 8 * 1000000},
    ]
})
register_benchmark({
    u'name': u'RoboschoolHarder',
    u'description': u'Test your might!!! Up to 12 hours on 32 cores',
    u'tasks': [
        {u'env_id': u"RoboschoolHumanoid-v1", u'trials': 4, u'num_timesteps': 100 * 1000000},
        {u'env_id': u"RoboschoolHumanoidFlagrun-v1", u'trials': 4, u'num_timesteps': 200 * 1000000},
        {u'env_id': u"RoboschoolHumanoidFlagrunHarder-v1", u'trials': 4, u'num_timesteps': 400 * 1000000},
    ]
})

# Other

_atari50 = [  # actually 47
    u'Alien', u'Amidar', u'Assault', u'Asterix', u'Asteroids',
    u'Atlantis', u'BankHeist', u'BattleZone', u'BeamRider', u'Bowling',
    u'Breakout', u'Centipede', u'ChopperCommand', u'CrazyClimber',
    u'DemonAttack', u'DoubleDunk', u'Enduro', u'FishingDerby', u'Freeway',
    u'Frostbite', u'Gopher', u'Gravitar', u'IceHockey', u'Jamesbond',
    u'Kangaroo', u'Krull', u'KungFuMaster', u'MontezumaRevenge', u'MsPacman',
    u'NameThisGame', u'Pitfall', u'Pong', u'PrivateEye', u'Qbert',
    u'RoadRunner', u'Robotank', u'Seaquest', u'SpaceInvaders', u'StarGunner',
    u'Tennis', u'TimePilot', u'Tutankham', u'UpNDown', u'Venture',
    u'VideoPinball', u'WizardOfWor', u'Zaxxon',
]

register_benchmark({
    u'name': u'Atari50_10M',
    u'description': u'47 Atari games from Mnih et al. (2013), with pixel observations, 10M timesteps',
    u'tasks': [{u'desc': _game, u'env_id': _game + _ATARI_SUFFIX, u'trials': 2, u'num_timesteps': int(10e6)} for _game in _atari50]
})

# HER DDPG

_fetch_tasks = [u'FetchReach-v1', u'FetchPush-v1', u'FetchSlide-v1']
register_benchmark({
    u'name': u'Fetch1M',
    u'description': u'Fetch* benchmarks for 1M timesteps',
    u'tasks': [{u'trials': 6, u'env_id': env_id, u'num_timesteps': int(1e6)} for env_id in _fetch_tasks]
})

