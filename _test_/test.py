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
import pynn.nn as nn
import time

_GRAD_CHECK_EPS = 1e-6
_BN_GRAD_CHECK_EPS = 1e-5
_FDIFF_EPS = 1e-8

_TEMP_FILE_NAME = '_temp_.pdata'

def vec_str(v):
    s = '[ '
    for i in range(len(v)):
        s += '%11.8f ' % v[i]
    s += ']'
    return s

def test_vec_pair(v1, msg1, v2, msg2, eps=_GRAD_CHECK_EPS):
    print msg1 + ' : ' + vec_str(v1)
    print msg2 + ' : ' + vec_str(v2)
    n_space = len(msg2) - len('diff')
    print ' ' * n_space + 'diff' + ' : ' + vec_str(v1 - v2)
    # err = np.sqrt(((v1 - v2)**2).sum())
    err = np.max(np.abs(v1 - v2))
    print 'err : %.8f' % err

    success = err < eps
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

def fdiff_grad_generator(net, x, t, add_noise=False, seed=None):
    if t is not None:
        net.load_target(t)

    def f(w):
        if add_noise and seed is not None:
            gnp.seed_rand(seed)
        w_0 = net.get_param_vec()
        net.set_param_from_vec(w)
        net.forward_prop(x, add_noise=add_noise, compute_loss=True)
        loss = net.get_loss()
        net.set_param_from_vec(w_0)

        return loss

    return f

def test_net_io(f_create, f_create_void):
    net1 = f_create()
    print 'Testing %s I/O' % net1.__class__.__name__

    net1.save_model_to_file(_TEMP_FILE_NAME)

    net2 = f_create_void()
    net2.load_model_from_file(_TEMP_FILE_NAME)

    os.remove(_TEMP_FILE_NAME)

    print 'Net #1: \n' + str(net1)
    print 'Net #2: \n' + str(net2)
    test_passed = (str(net1) == str(net2))

    test_passed = test_passed and test_vec_pair(net1.get_param_vec(), 'Net #1',
            net2.get_param_vec(), 'Net #2')
    return test_passed

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

def test_nonlin_invert(nonlin):
    print 'Testing inverting nonlinearity <%s>' % nonlin.get_name()

    sx, sy = 3, 4

    x = gnp.rand(sx, sy)
    y = nonlin.forward_prop(x)
    xx = nonlin.invert_output(y)

    test_passed = test_vec_pair(x.asarray().ravel(), '%15s' % 'Input',
            xx.asarray().ravel(), '%15s' % 'Inferred Input')
    print ''
    return test_passed

def test_all_nonlin():
    print ''
    print '=========================='
    print 'Testing all nonlinearities'
    print '=========================='
    print ''

    n_tests = 0
    n_success = 0

    for nonlin in ly.NONLIN_LIST:
        if test_nonlin(nonlin):
            n_success += 1
        if test_nonlin_invert(nonlin):
            n_success += 1
        n_tests += 2

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def test_loss(loss, weight=1):
    print 'Testing loss <%s>, weight=%g' % (loss.get_name(), weight)

    loss.set_weight(weight)

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
    elif loss.target_should_be_hinge():
        new_t = -np.ones(t.shape)
        new_t[np.arange(t.shape[0]), t.argmax(axis=1)] = 1
        t = gnp.garray(new_t)

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

    n_tests = len(ls.LOSS_LIST) * 2
    n_success = 0

    for loss in ls.LOSS_LIST:
        if test_loss(loss, weight=1):
            n_success += 1
        if test_loss(loss, weight=0.5):
            n_success += 1

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def test_layer(add_noise=False, no_loss=False, loss_after_nonlin=False,
        sparsity_weight=0, use_batch_normalization=False):
    print 'Testing layer ' + ('with noise' if add_noise else 'without noise') \
            + ', ' + ('without loss' if no_loss else 'with loss') \
            + ', ' + ('without sparsity' if sparsity_weight == 0 else 'with sparsity') \
            + ', ' + ('without batch normalization' if not use_batch_normalization else 'with batch normalization')
    in_dim = 4
    out_dim = 3
    n_cases = 3

    sparsity = 0.1

    x = gnp.randn(n_cases, in_dim)
    t = gnp.randn(n_cases, out_dim)

    if no_loss:
        loss = None
    else:
        loss = ls.get_loss_from_type_name(ls.LOSS_NAME_SQUARED)
        loss.load_target(t)
        loss.set_weight(2.5)

    seed = 8
    dropout_rate = 0.5 if add_noise else 0
    nonlin_type = ly.NONLIN_NAME_SIGMOID if sparsity_weight > 0 \
            else ly.NONLIN_NAME_TANH

    layer = ly.Layer(in_dim, out_dim, nonlin_type=nonlin_type,
            dropout=dropout_rate, sparsity=sparsity, sparsity_weight=sparsity_weight,
            loss=loss, loss_after_nonlin=loss_after_nonlin, use_batch_normalization=use_batch_normalization)

    if sparsity_weight > 0:
        # disable smoothing over minibatches
        layer._sparsity_smoothing = 1.0

    w_0 = layer.params.get_param_vec()

    if add_noise:
        gnp.seed_rand(seed)
    layer.params.clear_gradient()
    layer.forward_prop(x, compute_loss=True)
    layer.backward_prop()
    backprop_grad = layer.params.get_grad_vec()

    def f(w):
        if add_noise:
            # this makes sure the same units are dropped out every time this
            # function is called
            gnp.seed_rand(seed)
        layer.params.set_param_from_vec(w)
        layer.forward_prop(x, compute_loss=True)
        if layer.sparsity_weight == 0:
            return layer.loss_value
        else:
            return layer.loss_value + layer._sparsity_objective

    fdiff_grad = finite_difference_gradient(f, w_0)

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient', 
            eps=_GRAD_CHECK_EPS if not use_batch_normalization else _BN_GRAD_CHECK_EPS)
    print ''
    gnp.seed_rand(int(time.time()))
    return test_passed

def test_batch_normalization_layer():
    print 'Testing Batch Normalization layer'
    in_dim = 3
    n_cases = 5

    x = gnp.randn(n_cases, in_dim) * 2 + 3
    t = gnp.randn(n_cases, in_dim) * 2

    loss = ls.get_loss_from_type_name(ls.LOSS_NAME_SQUARED)
    loss.load_target(t)

    bn_layer = ly.BatchNormalizationLayer(in_dim)
    bn_layer.params.gamma = gnp.rand(in_dim)
    bn_layer.params.beta = gnp.rand(in_dim)

    w_0 = bn_layer.params.get_param_vec()

    y = bn_layer.forward_prop(x)
    _, loss_grad = loss.compute_not_weighted_loss_and_grad(y, True)
    bn_layer.backward_prop(loss_grad)

    backprop_grad = bn_layer.params.get_grad_vec()

    def f(w):
        bn_layer.params.set_param_from_vec(w)
        y = bn_layer.forward_prop(x)
        return loss.compute_not_weighted_loss_and_grad(y)[0]

    fdiff_grad = finite_difference_gradient(f, w_0)

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient', eps=_BN_GRAD_CHECK_EPS)
    print ''
    return test_passed

def test_all_layer():
    print ''
    print '==============='
    print 'Testing a layer'
    print '==============='
    print ''

    n_success = 0
    if test_layer(add_noise=False):
        n_success += 1
    if test_layer(add_noise=True):
        n_success += 1
    if test_layer(add_noise=False, no_loss=True):
        n_success += 1
    if test_layer(add_noise=False, loss_after_nonlin=True):
        n_success += 1
    if test_layer(add_noise=True, loss_after_nonlin=True):
        n_success += 1
    if test_layer(no_loss=True, sparsity_weight=1.0):
        n_success += 1
    if test_layer(sparsity_weight=1.0):
        n_success += 1
    if test_layer(add_noise=True, sparsity_weight=1.0):
        n_success += 1
    if test_batch_normalization_layer():
        n_success += 1
    if test_layer(add_noise=True, use_batch_normalization=True):
        n_success += 1
    if test_layer(add_noise=False, use_batch_normalization=True):
        n_success += 1
    if test_layer(add_noise=True, use_batch_normalization=True, loss_after_nonlin=True):
        n_success += 1
    if test_layer(add_noise=False, use_batch_normalization=True, loss_after_nonlin=False):
        n_success += 1
    if test_layer(add_noise=True, use_batch_normalization=True, loss_after_nonlin=True, sparsity_weight=1.0):
        n_success += 1

    n_tests = 14

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def create_neuralnet(dropout_rate, loss_after_nonlin=False, use_batch_normalization=False):
    in_dim = 3
    out_dim = 2
    h1_dim = 2
    h2_dim = 2
    h3_dim = 2

    net = nn.NeuralNet(in_dim, out_dim)
    net.add_layer(h1_dim, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)
    net.add_layer(h2_dim, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate,
            use_batch_normalization=use_batch_normalization)
    #net.add_layer(h3_dim, nonlin_type=ly.NONLIN_NAME_RELU, dropout=dropout_rate)
    #net.add_layer(10, nonlin_type=ly.NONLIN_NAME_RELU, dropout=dropout_rate)
    #net.add_layer(10, nonlin_type=ly.NONLIN_NAME_RELU, dropout=dropout_rate)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=dropout_rate,
            use_batch_normalization=use_batch_normalization)

    net.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=1.1, loss_after_nonlin=loss_after_nonlin)
    return net

def test_neuralnet(add_noise=False, loss_after_nonlin=False, use_batch_normalization=False):
    print 'Testing NeuralNet, ' + ('with noise' if add_noise else 'without noise') \
            + ', ' + ('with BN' if use_batch_normalization else 'without BN')
    n_cases = 5
    seed = 8
    dropout_rate = 0.5 if add_noise else 0

    net = create_neuralnet(dropout_rate, loss_after_nonlin=loss_after_nonlin, use_batch_normalization=use_batch_normalization)
   
    print net

    x = gnp.randn(n_cases, net.in_dim)
    t = gnp.randn(n_cases, net.out_dim)

    if net.loss.target_should_be_one_hot():
        new_t = np.zeros(t.shape)
        new_t[np.arange(t.shape[0]), t.argmax(axis=1)] = 1
        t = gnp.garray(new_t)
    elif net.loss.target_should_be_normalized():
        t = t - t.min(axis=1)[:,gnp.newaxis] + 1
        t /= t.sum(axis=1)[:,gnp.newaxis]

    net.load_target(t)

    if add_noise:
        gnp.seed_rand(seed)
    net.forward_prop(x, add_noise=add_noise, compute_loss=True)
    net.clear_gradient()
    net.backward_prop()

    backprop_grad = net.get_grad_vec()

    f = fdiff_grad_generator(net, x, None, add_noise=add_noise, seed=seed)
    fdiff_grad = finite_difference_gradient(f, net.get_param_vec())

    eps = _BN_GRAD_CHECK_EPS if use_batch_normalization else _GRAD_CHECK_EPS

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient', eps=eps)
    print ''

    gnp.seed_rand(int(time.time()))
    return test_passed

def test_neuralnet_io(loss_after_nonlin=False, use_batch_normalization=False):
    def f_create():
        return create_neuralnet(0.5, loss_after_nonlin=loss_after_nonlin, use_batch_normalization=use_batch_normalization)
    def f_create_void():
        return nn.NeuralNet(0,0)
    return test_net_io(f_create, f_create_void)

def test_all_neuralnet():
    print ''
    print '================='
    print 'Testing NeuralNet'
    print '================='
    print ''

    n_success = 0
    if test_neuralnet(add_noise=False):
        n_success += 1
    if test_neuralnet(add_noise=True):
        n_success += 1
    if test_neuralnet(add_noise=False, loss_after_nonlin=True):
        n_success += 1
    if test_neuralnet(add_noise=True, loss_after_nonlin=True):
        n_success += 1
    if test_neuralnet(add_noise=False, loss_after_nonlin=True, use_batch_normalization=True):
        n_success += 1
    if test_neuralnet(add_noise=True, loss_after_nonlin=True, use_batch_normalization=True):
        n_success += 1
    if test_neuralnet_io():
        n_success += 1
    if test_neuralnet_io(loss_after_nonlin=True):
        n_success += 1
    if test_neuralnet_io(loss_after_nonlin=True, use_batch_normalization=True):
        n_success += 1

    n_tests = 9

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def create_stacked_net(in_dim, out_dim, dropout_rate):
    net1 = nn.NeuralNet(3,out_dim[0])
    net1.add_layer(2, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)
    net1.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
    net1.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=0.5)

    net2 = nn.NeuralNet(out_dim[0], out_dim[1])
    net2.add_layer(3, nonlin_type=ly.NONLIN_NAME_RELU, dropout=dropout_rate)
    net2.add_layer(0, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)

    net3 = nn.NeuralNet(out_dim[1], out_dim[2])
    net3.add_layer(1, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
    net3.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=0)
    net3.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=1)

    return nn.StackedNeuralNet(net1, net2, net3)

def test_stacked_net_gradient(add_noise=False):
    print 'Testing StackedNeuralNet'

    in_dim = 3
    out_dim = [5, 2, 2]
    n_cases = 5
    seed = 8
    dropout_rate = 0.5 if add_noise else 0

    stacked_net = create_stacked_net(in_dim, out_dim, dropout_rate)

    print stacked_net

    x = gnp.randn(n_cases, in_dim)
    t1 = gnp.randn(n_cases, out_dim[0])
    t3 = gnp.randn(n_cases, out_dim[2])

    stacked_net.load_target(t1, None, t3)

    if add_noise:
        gnp.seed_rand(seed)

    stacked_net.clear_gradient()
    stacked_net.forward_prop(x, add_noise=add_noise, compute_loss=True)
    stacked_net.backward_prop()

    backprop_grad = stacked_net.get_grad_vec()

    f = fdiff_grad_generator(stacked_net, x, None, add_noise=add_noise, seed=seed)
    fdiff_grad = finite_difference_gradient(f, stacked_net.get_param_vec())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''

    gnp.seed_rand(int(time.time()))
    return test_passed

def test_stacked_net_io():
    def f_create():
        return create_stacked_net(3, [5,2,2], 0.5)
    def f_create_void():
        return nn.StackedNeuralNet()
    return test_net_io(f_create, f_create_void)

def test_all_stacked_net():
    print ''
    print '========================'
    print 'Testing StackedNeuralNet'
    print '========================'
    print ''

    n_success = 0
    if test_stacked_net_gradient(add_noise=False):
        n_success += 1
    if test_stacked_net_gradient(add_noise=True):
        n_success += 1
    if test_stacked_net_io():
        n_success += 1

    n_tests = 3

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def create_y_net(in_dim, out_dim, dropout_rate):

    net01 = nn.NeuralNet(3,2)
    net01.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
    net02 = nn.NeuralNet(2, out_dim[0])
    net02.add_layer(0, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout_rate)
    net02.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=0.5)
    net1 = nn.StackedNeuralNet(net01, net02)

    net2 = nn.NeuralNet(out_dim[0], out_dim[1])
    net2.add_layer(0, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)
    net2.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=0)

    net3 = nn.NeuralNet(out_dim[0], out_dim[2])
    net3.add_layer(1, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
    net3.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=0)
    net3.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=1.5)

    ynet = nn.YNeuralNet(net1, net2, net3)

    return ynet

def test_y_net_gradient(add_noise=False):
    print 'Testing YNeuralNet ' + ('with noise' if add_noise else 'without noise')

    in_dim = 3
    out_dim = [2, 2, 2]
    n_cases = 5
    seed = 8
    dropout_rate = 0.5 if add_noise else 0

    ynet = create_y_net(in_dim, out_dim, dropout_rate)

    print ynet 

    x = gnp.randn(n_cases, in_dim)
    t1 = gnp.randn(n_cases, out_dim[0])
    t2 = gnp.randn(n_cases, out_dim[1])
    t3 = gnp.randn(n_cases, out_dim[2])

    ynet.load_target([None, t1], t2, t3)

    if add_noise:
        gnp.seed_rand(seed)

    ynet.clear_gradient()
    ynet.forward_prop(x, add_noise=add_noise, compute_loss=True)
    ynet.backward_prop()

    backprop_grad = ynet.get_grad_vec()

    f = fdiff_grad_generator(ynet, x, None, add_noise=add_noise, seed=seed)
    fdiff_grad = finite_difference_gradient(f, ynet.get_param_vec())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''

    gnp.seed_rand(int(time.time()))
    return test_passed

def test_y_net_io():
    def f_create():
        in_dim = 3
        out_dim = [2, 2, 2]
        n_cases = 5
        seed = 8

        dropout_rate = 0.5

        net1 = nn.NeuralNet(3,2)
        net1.add_layer(2, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
        net1.add_layer(0, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout_rate)
        net1.set_loss(ls.LOSS_NAME_SQUARED)

        net2 = nn.NeuralNet(out_dim[0], out_dim[1])
        net2.add_layer(0, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)
        net2.set_loss(ls.LOSS_NAME_SQUARED)

        net3 = nn.NeuralNet(out_dim[0], out_dim[2])
        net3.add_layer(1, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
        net3.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=0)
        net3.set_loss(ls.LOSS_NAME_SQUARED)
        return nn.YNeuralNet(net1, net2, net3)
    def f_create_void():
        return nn.YNeuralNet()
    return test_net_io(f_create, f_create_void)

def test_all_y_net():
    print ''
    print '=================='
    print 'Testing YNeuralNet'
    print '=================='
    print ''

    n_success = 0
    if test_y_net_gradient(add_noise=False):
        n_success += 1
    if test_y_net_gradient(add_noise=True):
        n_success += 1
    if test_y_net_io():
        n_success += 1

    n_tests = 3

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def create_autoencoder(dropout_rate=0):
    in_dim = 3
    h_dim = 2

    net1 = nn.NeuralNet(in_dim, h_dim)
    net1.add_layer(2, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)
    net1.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=dropout_rate)
    net2 = nn.NeuralNet(h_dim, in_dim)
    net2.add_layer(2, nonlin_type=ly.NONLIN_NAME_TANH, dropout=0)
    net2.add_layer(1, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout_rate)
    net2.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=dropout_rate)
    net2.set_loss(ls.LOSS_NAME_SQUARED, loss_weight=1.5)

    autoencoder = nn.AutoEncoder(net1, net2)
    return autoencoder

def test_autoencoder(add_noise=False):
    print 'Testing AutoEncoder ' + ('with noise' if add_noise else 'without noise')
    n_cases = 5
    seed = 8

    dropout_rate = 0.5 if add_noise else 0

    autoencoder = create_autoencoder(dropout_rate)
    print autoencoder

    x = gnp.randn(n_cases, autoencoder.in_dim)

    if add_noise:
        gnp.seed_rand(seed)

    autoencoder.clear_gradient()
    autoencoder.forward_prop(x, add_noise=add_noise, compute_loss=True)
    autoencoder.backward_prop()

    backprop_grad = autoencoder.get_grad_vec()

    f = fdiff_grad_generator(autoencoder, x, None, add_noise=add_noise, seed=seed)
    fdiff_grad = finite_difference_gradient(f, autoencoder.get_param_vec())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''

    gnp.seed_rand(int(time.time()))
    return test_passed

def test_autoencoder_io():
    def f_create():
        return create_autoencoder(0.5)
    def f_create_void():
        return nn.AutoEncoder()
    return test_net_io(f_create, f_create_void)

def test_all_autoencoder():
    print ''
    print '==================='
    print 'Testing AutoEncoder'
    print '==================='
    print ''

    n_success = 0
    if test_autoencoder(add_noise=False):
        n_success += 1
    if test_autoencoder(add_noise=True):
        n_success += 1
    if test_autoencoder_io():
        n_success += 1

    n_tests = 3

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def run_all_tests():
    gnp.seed_rand(int(time.time()))

    n_success = 0
    n_tests = 0

    test_list = [test_all_nonlin, test_all_loss, test_all_layer,
            test_all_neuralnet, test_all_stacked_net, test_all_y_net, 
            test_all_autoencoder]
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

