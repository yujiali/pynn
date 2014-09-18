"""
Test the neural network package.

Yujia Li, 09/2014
"""

import os
os.environ['GNUMPY_CPU_PRECISION'] = '64'

import gnumpy as gnp
import numpy as np
import pynn.layer as ly
import pynn.loss as ls
import time

_GRAD_CHECK_EPS = 1e-6
_FDIFF_EPS = 1e-8

def vec_str(v):
    s = '[ '
    for i in range(len(v)):
        s += '%11.8f ' % v[i]
    s += ']'
    return s

def test_vec_pair(v1, msg1, v2, msg2):
    print msg1 + ' : ' + vec_str(v1)
    print msg2 + ' : ' + vec_str(v2)
    n_space = len(msg2) - len('diff')
    print ' ' * n_space + 'diff' + ' : ' + vec_str(v1 - v2)
    err = np.sqrt(((v1 - v2)**2).sum())
    print 'err : %.8f' % err

    success = err < _GRAD_CHECK_EPS
    print '** SUCCESS **' if success else '** FAIL **'

    return success

def finite_difference_gradient(f, x):
    grad = x * 0
    for i in range(len(x)):
        x_0 = x[i]
        x[i] = x_0 + _FDIFF_EPS
        f_plus = f(x)
        x[i] = x_0 - _FDIFF_EPS
        f_minus = f(x)
        grad[i] = (f_plus - f_minus) / (2 * _FDIFF_EPS)
        x[i] = x_0

    return grad

def test_nonlin(nonlin):
    print 'Testing nonlinearity <%s>' % nonlin.get_name()

    sx, sy = 3, 4

    def f(w):
        return nonlin.forward_prop(gnp.garray(w.reshape(sx, sy))).sum()

    x = gnp.randn(sx, sy)
    y = nonlin.forward_prop(x)

    fdiff_grad = finite_difference_gradient(f, x.asarray().ravel())
    backprop_grad = nonlin.backward_prop(x, y).asarray().ravel()

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_all_nonlin():
    print ''
    print '=========================='
    print 'Testing all nonlinearities'
    print '=========================='
    print ''

    n_tests = len(ly.NONLIN_LIST)
    n_success = 0

    for nonlin in ly.NONLIN_LIST:
        if test_nonlin(nonlin):
            n_success += 1

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def test_loss(loss):
    print 'Testing loss <%s>' % loss.get_name()

    sx, sy = 3, 4

    x = gnp.randn(sx, sy)
    t = gnp.randn(sx, sy)

    if loss.target_should_be_one_hot():
        new_t = np.zeros(t.shape)
        new_t[np.arange(t.shape[0]), t.argmax(axis=1)] = 1
        t = gnp.garray(new_t)
    elif loss.target_should_be_normalized():
        t = t - t.min(axis=1)[:,gnp.newaxis] + 1
        t /= t.sum(axis=1)[:,gnp.newaxis]

    print t
        
    loss.load_target(t)

    def f(w):
        return loss.compute_loss_and_grad(gnp.garray(w.reshape(sx, sy)))[0]

    fdiff_grad = finite_difference_gradient(f, x.asarray().ravel())
    backprop_grad = loss.compute_loss_and_grad(x, compute_grad=True)[1].asarray().ravel()

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_all_loss():
    print ''
    print '=================='
    print 'Testing all losses'
    print '=================='
    print ''

    n_tests = len(ls.LOSS_LIST)
    n_success = 0

    for loss in ls.LOSS_LIST:
        if test_loss(loss):
            n_success += 1

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def run_all_tests():
    gnp.seed_rand(int(time.time()))

    n_success = 0
    n_tests = 0

    test_list = [test_all_nonlin, test_all_loss]
    for batch_test in test_list:
        success_in_batch, tests_in_batch = batch_test()
        n_success += success_in_batch
        n_tests += tests_in_batch

    print ''
    print '==================='
    print 'All tests finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''


if __name__ == '__main__':
    run_all_tests()

