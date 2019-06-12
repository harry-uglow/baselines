from __future__ import absolute_import
import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print u'This Python is only compatible with Python 3, but you are running '
          u'Python {}. The installation will likely fail.'.format(sys.version_info.major)


extras = {
    u'test': [
        u'filelock',
        u'pytest',
        u'pytest-forked',
        u'atari-py'
    ],
    u'bullet': [
        u'pybullet',
    ],
    u'mpi': [
        u'mpi4py'
    ]
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras[u'all'] = all_deps

setup(name=u'baselines',
      packages=[package for package in find_packages()
                if package.startswith(u'baselines')],
      install_requires=[
          u'gym',
          u'scipy',
          u'tqdm',
          u'joblib',
          u'dill',
          u'progressbar2',
          u'cloudpickle',
          u'click',
          u'opencv-python'
      ],
      extras_require=extras,
      description=u'OpenAI baselines: high quality implementations of reinforcement learning algorithms',
      author=u'OpenAI',
      url=u'https://github.com/openai/baselines',
      author_email=u'gym@openai.com',
      version=u'0.1.5')


# ensure there is some tensorflow build with version above 1.4
import pkg_resources
tf_pkg = None
for tf_pkg_name in [u'tensorflow', u'tensorflow-gpu', u'tf-nightly', u'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, u'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion
assert LooseVersion(re.sub(ur'-?rc\d+$', u'', tf_pkg.version)) >= LooseVersion(u'1.4.0')
