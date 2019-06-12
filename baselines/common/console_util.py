from __future__ import absolute_import
from contextlib import contextmanager
import numpy as np
import time
import shlex
import subprocess

# ================================================================
# Misc
# ================================================================

def fmt_row(width, row, header=False):
    out = u" | ".join(fmt_item(x, width) for x in row)
    if header: out = out + u"\n" + u"-"*len(out)
    return out

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, (float, np.float32, np.float64)):
        v = abs(x)
        if (v < 1e-4 or v > 1e+4) and v > 0:
            rep = u"%7.2e" % x
        else:
            rep = u"%7.5f" % x
    else: rep = unicode(x)
    return u" "*(l - len(rep)) + rep

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color=u'green', bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append(u'1')
    return u'\x1b[%sm%s\x1b[0m' % (u';'.join(attr), string)

def print_cmd(cmd, dry=False):
    if isinstance(cmd, unicode):  # for shell=True
        pass
    else:
        cmd = u' '.join(shlex.quote(arg) for arg in cmd)
    x = colorize((u'CMD: ' if not dry else u'DRY: ') + cmd)
    print x


def get_git_commit(cwd=None):
    return subprocess.check_output([u'git', u'rev-parse', u'--short', u'HEAD'], cwd=cwd).decode(u'utf8')

def get_git_commit_message(cwd=None):
    return subprocess.check_output([u'git', u'show', u'-s', u'--format=%B', u'HEAD'], cwd=cwd).decode(u'utf8')

def ccap(cmd, dry=False, env=None, **kwargs):
    print_cmd(cmd, dry)
    if not dry:
        subprocess.check_call(cmd, env=env, **kwargs)


MESSAGE_DEPTH = 0

@contextmanager
def timed(msg):
    global MESSAGE_DEPTH #pylint: disable=W0603
    print colorize(u'\t'*MESSAGE_DEPTH + u'=: ' + msg, color=u'magenta')
    tstart = time.time()
    MESSAGE_DEPTH += 1
    yield
    MESSAGE_DEPTH -= 1
    print colorize(u'\t'*MESSAGE_DEPTH + u"done in %.3f seconds"%(time.time() - tstart), color=u'magenta')
