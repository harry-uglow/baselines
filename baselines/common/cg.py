from __future__ import division
from __future__ import absolute_import
import numpy as np
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    u"""
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  u"%10i %10.3g %10.3g"
    titlestr =  u"%10s %10s %10s"
    if verbose: print titlestr % (u"iter", u"residual norm", u"soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr % (i+1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x
