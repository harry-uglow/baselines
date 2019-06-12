from __future__ import absolute_import
import os, subprocess, sys

def mpi_fork(n, bind_to_core=False):
    u"""Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1:
        return u"child"
    if os.getenv(u"IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS=u"1",
            OMP_NUM_THREADS=u"1",
            IN_MPI=u"1"
        )
        args = [u"mpirun", u"-np", unicode(n)]
        if bind_to_core:
            args += [u"-bind-to", u"core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        return u"parent"
    else:
        return u"child"
