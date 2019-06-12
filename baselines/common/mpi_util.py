from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from collections import defaultdict
import os, numpy as np
import platform
import shutil
import subprocess
import warnings
import sys
from itertools import izip
from itertools import imap
from io import open

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def sync_from_root(sess, variables, comm=None):
    u"""
    Send the root node's parameters to every worker.
    Arguments:
      sess: the TensorFlow session.
      variables: all parameter variables including optimizer's
    """
    if comm is None: comm = MPI.COMM_WORLD
    import tensorflow as tf
    values = comm.bcast(sess.run(variables))
    sess.run([tf.assign(var, val)
        for (var, val) in izip(variables, values)])

def gpu_count():
    u"""
    Count the GPUs on this machine.
    """
    if shutil.which(u'nvidia-smi') is None:
        return 0
    output = subprocess.check_output([u'nvidia-smi', u'--query-gpu=gpu_name', u'--format=csv'])
    return max(0, len(output.split('\n')) - 2)

def setup_mpi_gpus():
    u"""
    Set CUDA_VISIBLE_DEVICES to MPI rank if not already set
    """
    if u'CUDA_VISIBLE_DEVICES' not in os.environ:
        if sys.platform == u'darwin': # This Assumes if you're on OSX you're just
            ids = []                 # doing a smoke test and don't want GPUs
        else:
            lrank, _lsize = get_local_rank_size(MPI.COMM_WORLD)
            ids = [lrank]
        os.environ[u"CUDA_VISIBLE_DEVICES"] = u",".join(imap(unicode, ids))

def get_local_rank_size(comm):
    u"""
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.

    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]

def share_file(comm, path):
    u"""
    Copies the file from rank 0 to all other ranks
    Puts it in the same place on all machines
    """
    localrank, _ = get_local_rank_size(comm)
    if comm.Get_rank() == 0:
        with open(path, u'rb') as fh:
            data = fh.read()
        comm.bcast(data)
    else:
        data = comm.bcast(None)
        if localrank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, u'wb') as fh:
                fh.write(data)
    comm.Barrier()

def dict_gather(comm, d, op=u'mean', assert_all_have_data=True):
    u"""
    Perform a reduction operation over dicts
    """
    if comm is None: return d
    alldicts = comm.allgather(d)
    size = comm.size
    k2li = defaultdict(list)
    for d in alldicts:
        for (k,v) in d.items():
            k2li[k].append(v)
    result = {}
    for (k,li) in k2li.items():
        if assert_all_have_data:
            assert len(li)==size, u"only %i out of %i MPI workers have sent '%s'" % (len(li), size, k)
        if op==u'mean':
            result[k] = np.mean(li, axis=0)
        elif op==u'sum':
            result[k] = np.sum(li, axis=0)
        else:
            assert 0, op
    return result

def mpi_weighted_mean(comm, local_name2valcount):
    u"""
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(u'WARNING: tried to compute mean on non-float {}={}'.format(name, val))
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return dict((name, name2sum[name] / name2count[name]) for name in name2sum)
    else:
        return {}

