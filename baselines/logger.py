from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from io import open
from itertools import imap
from itertools import ifilter

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, unicode):
            self.file = open(filename_or_file, u'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, u'read'), u'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = u'%-8.3g' % (val,)
            else:
                valstr = unicode(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print u'WARNING: tried to write empty key-value dict'
            return
        else:
            keywidth = max(imap(len, key2str.keys()))
            valwidth = max(imap(len, key2str.values()))

        # Write out the data
        dashes = u'-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(u'| %s%s | %s%s |' % (
                key,
                u' ' * (keywidth - len(key)),
                val,
                u' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write(u'\n'.join(lines) + u'\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[:maxlen-3] + u'...' if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1: # add space unless this is the last one
                self.file.write(u' ')
        self.file.write(u'\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, u'wt')

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, u'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + u'\n')
        self.file.flush()

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, u'w+t')
        self.keys = []
        self.sep = u','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(u',')
                self.file.write(k)
            self.file.write(u'\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write(u'\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(u',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(unicode(v))
        self.file.write(u'\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    u"""
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = u'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {u'tag': k, u'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

def make_output_format(format, ev_dir, log_suffix=u''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == u'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == u'log':
        return HumanOutputFormat(osp.join(ev_dir, u'log%s.txt' % log_suffix))
    elif format == u'json':
        return JSONOutputFormat(osp.join(ev_dir, u'progress%s.json' % log_suffix))
    elif format == u'csv':
        return CSVOutputFormat(osp.join(ev_dir, u'progress%s.csv' % log_suffix))
    elif format == u'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, u'tb%s' % log_suffix))
    else:
        raise ValueError(u'Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def logkv(key, val):
    u"""
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)

def logkv_mean(key, val):
    u"""
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)

def logkvs(d):
    u"""
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def dumpkvs():
    u"""
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()

def getkvs():
    return get_current().name2val


def log(*args, **_3to2kwargs):
    if 'level' in _3to2kwargs: level = _3to2kwargs['level']; del _3to2kwargs['level']
    else: level = INFO
    u"""
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    u"""
    Set logging threshold on current logger.
    """
    get_current().set_level(level)

def set_comm(comm):
    get_current().set_comm(comm)

def get_dir():
    u"""
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

@contextmanager
def profile_kv(scopename):
    logkey = u'wait_' + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart

def profile(n):
    u"""
    Usage:
    @profile("my_func")
    def my_func(): code
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)
        return func_wrapper
    return decorator_with_name


# ================================================================
# Backend
# ================================================================

def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            from baselines.common import mpi_util
            d = mpi_util.mpi_weighted_mean(self.comm,
                dict((name, (val, self.name2cnt.get(name, 1)))
                    for (name, val) in self.name2val.items()))
            if self.comm.rank != 0:
                d[u'dummy'] = 1 # so we don't get a warning about empty dict
        out = d.copy() # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, **_3to2kwargs):
        if 'level' in _3to2kwargs: level = _3to2kwargs['level']; del _3to2kwargs['level']
        else: level = INFO
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(imap(unicode, args))

def configure(dir=None, format_strs=None, comm=None):
    u"""
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv(u'OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime(u"openai-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, unicode)
    os.makedirs(dir, exist_ok=True)

    log_suffix = u''
    rank = 0
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in [u'PMI_RANK', u'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            rank = int(os.environ[varname])
    if rank > 0:
        log_suffix = u"-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv(u'OPENAI_LOG_FORMAT', u'stdout,log,csv').split(u',')
        else:
            format_strs = os.getenv(u'OPENAI_LOG_FORMAT_MPI', u'log').split(u',')
    format_strs = ifilter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    log(u'Logging to %s'%dir)

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log(u'Reset logger')

@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

# ================================================================

def _demo():
    info(u"hi")
    debug(u"shouldn't appear")
    set_level(DEBUG)
    debug(u"should appear")
    dir = u"/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv(u"a", 3)
    logkv(u"b", 2.5)
    dumpkvs()
    logkv(u"b", -2.5)
    logkv(u"a", 5.5)
    dumpkvs()
    info(u"^^^ should see a = 5.5")
    logkv_mean(u"b", -22.5)
    logkv_mean(u"b", -44.4)
    logkv(u"a", 5.5)
    dumpkvs()
    info(u"^^^ should see b = -33.3")

    logkv(u"b", -2.5)
    dumpkvs()

    logkv(u"a", u"longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


# ================================================================
# Readers
# ================================================================

def read_json(fname):
    import pandas
    ds = []
    with open(fname, u'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=None, comment=u'#')

def read_tb(path):
    u"""
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    import tensorflow as tf
    if osp.isdir(path):
        fnames = glob(osp.join(path, u"events.*"))
    elif osp.basename(path).startswith(u"events."):
        fnames = [path]
    else:
        raise NotImplementedError(u"Expected tensorboard file or directory containing them. Got %s"%path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx,tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step-1, colidx] = value
    return pandas.DataFrame(data, columns=tags)

if __name__ == u"__main__":
    _demo()
