"""
Test the neural network package.

Yujia Li, 09/2014
"""

import numpy as np

_GRAD_CHECK_EPS = 1e-7

def vec_str(v):
    s = '[ ',
    for i in range(len(v)):
        s += '%11.8f ' % v[i]
    s += ']'
    return s

def test_vec_pair(v1, msg1, v2, msg2):
    print msg1 + ' : ' + vec_str(v1)
    print msg2 + ' : ' + vec_str(v2)
    err = np.sqrt(((v1 - v2)**2).sum())
    print 'err : %.8f' % err

    success = err < _GRAD_CHECK_EPS
    print '** SUCCESS **' if success else '** FAIL **'

    return success

