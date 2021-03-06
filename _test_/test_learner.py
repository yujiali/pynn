import cPickle as pickle

import pynn.learner as learner
import pynn.nn as nn
import pynn.layer as ly
import pynn.loss as ls
import numpy as np

def load_toy_data():
    import os
    with open(os.path.dirname(os.path.realpath(__file__)) + '/mnist_toy_100.pdata', 'rb') as f:
        d = pickle.load(f)

    t = d['t_train']
    K = t.max() + 1

    t_train = np.zeros((len(t), K))
    t_train[np.arange(len(t)), t] = 1

    t = d['t_val']

    t_val = np.zeros((len(t), K))
    t_val[np.arange(len(t)), t] = 1

    return d['x_train'], t_train, d['x_val'], t_val

def build_classification_net(in_dim, out_dim, dropout=0, use_batch_normalization=False):
    net = nn.NeuralNet(in_dim, out_dim)
    net.add_layer(128, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout, use_batch_normalization=use_batch_normalization)
    net.add_layer(32, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout, use_batch_normalization=use_batch_normalization)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=dropout)
    net.set_loss(ls.LOSS_NAME_CROSSENTROPY)

    return net

def test_neural_net_learner():
    x_train, t_train, x_val, t_val = load_toy_data()
    print 'Data loaded'

    in_dim = x_train.shape[1]
    out_dim = t_train.shape[1]

    dropout = 0.2
    learn_rate = 1e-1
    momentum = 0.5
    max_iters = 500

    net = build_classification_net(in_dim, out_dim, dropout=dropout)
    print 'Network #1 constructed: ' + str(net)

    # nn_learner = learner.Learner(net)
    nn_learner = learner.ClassificationLearner(net, param_cache_size=1)
    nn_learner.load_data(x_train, t_train, x_val, t_val)
    nn_learner.train_gradient_descent(learn_rate=learn_rate, momentum=momentum, 
        weight_decay=0, learn_rate_schedule=None, momentum_schedule=None,
        learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
        max_iters=max_iters, iprint=10, verbose=True)

    net = build_classification_net(in_dim, out_dim, use_batch_normalization=True)
    print 'Network #2 constructed: ' + str(net)

    # nn_learner = learner.Learner(net)
    nn_learner = learner.ClassificationLearner(net, param_cache_size=1)
    nn_learner.load_data(x_train, t_train, x_val, t_val)
    nn_learner.train_gradient_descent(learn_rate=learn_rate, momentum=momentum, 
        weight_decay=0, learn_rate_schedule=None, momentum_schedule=None,
        learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
        max_iters=max_iters, iprint=10, verbose=True)

def test_minibatch_generator():
    import gnumpy as gnp
    x = np.arange(30).reshape(10,3)
    t = np.arange(10)

    mbgen = learner.MiniBatchGenerator(x, t=t, minibatch_size=3, random_order=False)
    for i in range(5):
        print ''
        print mbgen.next()

    mbgen.shuffle_data()
    for i in range(4):
        print ''
        print mbgen.next()

    print '======================='
    mbgen = learner.MiniBatchGenerator(
            gnp.garray(x), t=t, minibatch_size=3, random_order=True)
    for i in range(5):
        print ''
        print mbgen.next()

    mbgen.shuffle_data()
    for i in range(4):
        print ''
        print mbgen.next()

def test_neural_net_sgd_learner():
    x_train, t_train, x_val, t_val = load_toy_data()
    print 'Data loaded'

    in_dim = x_train.shape[1]
    out_dim = t_train.shape[1]

    minibatch_size=10
    max_iters=500
    dropout = 0.2
    learn_rate = 1e-2
    momentum = 0.9

    net = build_classification_net(in_dim, out_dim, dropout=dropout)
    print 'Network #1 constructed: ' + str(net)

    # nn_learner = learner.Learner(net)
    nn_learner = learner.ClassificationLearner(net, param_cache_size=1)
    nn_learner.load_data(x_train, t_train, x_val, t_val)
    nn_learner.train_sgd(minibatch_size=minibatch_size, learn_rate=learn_rate, momentum=momentum, 
        weight_decay=0, learn_rate_schedule=None, momentum_schedule=None,
        learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
        max_iters=max_iters, iprint=10, verbose=True)
    
    print ''
    print ''

    net = build_classification_net(in_dim, out_dim, dropout=dropout, use_batch_normalization=True)
    print 'Network #2 constructed: ' + str(net)

    # nn_learner = learner.Learner(net)
    nn_learner = learner.ClassificationLearner(net, param_cache_size=1)
    nn_learner.load_data(x_train, t_train, x_val, t_val)
    nn_learner.train_sgd(minibatch_size=minibatch_size, learn_rate=learn_rate, momentum=momentum, 
        weight_decay=0, learn_rate_schedule=None, momentum_schedule=None,
        learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
        max_iters=max_iters, iprint=10, verbose=True)

def test_autoencoder_pretraining():
    x_train, t_train, x_val, t_val = load_toy_data()

    in_dim = x_train.shape[1]
    h_dim = 5

    enc = nn.NeuralNet(in_dim, h_dim)
    enc.add_layer(30, nonlin_type=ly.NONLIN_NAME_SIGMOID, use_batch_normalization=True)
    enc.add_layer(20, nonlin_type=ly.NONLIN_NAME_TANH, use_batch_normalization=True)
    enc.add_layer(10, nonlin_type=ly.NONLIN_NAME_RELU, use_batch_normalization=True)
    enc.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR)

    dec = nn.NeuralNet(h_dim, in_dim)
    dec.add_layer(10, nonlin_type=ly.NONLIN_NAME_RELU)
    dec.add_layer(20, nonlin_type=ly.NONLIN_NAME_TANH)
    dec.add_layer(30, nonlin_type=ly.NONLIN_NAME_SIGMOID)
    dec.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID)

    ae = nn.AutoEncoder(enc, dec)

    print ''
    print ae
    print ''

    pretrainer = learner.AutoEncoderPretrainer(ae)
    pretrainer.load_data(x_train)
    pretrainer.pretrain_network(learn_rate=1e-1, momentum=0.5, weight_decay=0, 
            minibatch_size=10, max_grad_norm=10, max_iters=1000, iprint=50)

if __name__ == '__main__':
    test_neural_net_sgd_learner()
    #test_neural_net_learner()
    #test_minibatch_generator()
    #test_autoencoder_pretraining()

