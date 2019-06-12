from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from baselines import logger
from baselines.common.tests.test_with_mpi import with_mpi
from baselines.common import mpi_util

@with_mpi()
def test_mpi_weighted_mean():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    with logger.scoped_configure(comm=comm):
        if comm.rank == 0:
            name2valcount = {u'a' : (10, 2), u'b' : (20,3)}
        elif comm.rank == 1:
            name2valcount = {u'a' : (19, 1), u'c' : (42,3)}
        else:
            raise NotImplementedError

        d = mpi_util.mpi_weighted_mean(comm, name2valcount)
        correctval = {u'a' : (10 * 2 + 19) / 3.0, u'b' : 20, u'c' : 42}
        if comm.rank == 0:
            assert d == correctval, u'{} != {}'.format(d, correctval)

        for name, (val, count) in name2valcount.items():
            for _ in xrange(count):
                logger.logkv_mean(name, val)
        d2 = logger.dumpkvs()
        if comm.rank == 0:
            assert d2 == correctval
