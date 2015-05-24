"""
Test RNN training on some simple tasks.

Yujia Li, 05/2015
"""

import pynn.rnn as rnn
import pynn.layer as ly
import pynn.loss as ls
import pynn.nn as nn
import numpy as np

def num_to_bin_array(n):
    s = bin(n)[2:]
    x = np.zeros(len(s), dtype=np.float)
    for i in xrange(len(s)):
        x[i] = float(s[i])
    return x[::-1]

def bin_array_to_num(x):
    x = x[::-1]
    n = 0
    for i in xrange(len(x)):
        n = n * 2 + x[i]
    return n

def get_binary_matrix_for_pair(a, b):
    x0 = num_to_bin_array(a)
    x1 = num_to_bin_array(b)

    xx = np.zeros((max(len(x0), len(x1))+2, 2), dtype=np.float)
    xx[:x0.size,0] = x0
    xx[:x1.size,1] = x1

    return xx

def generate_binary_add_data(n, int_max=100):
    x_raw = np.random.randint(0, int_max, (n,2))
    t_raw = x_raw.sum(axis=1)

    x = []
    t = []

    for i in xrange(n):
        xx = get_binary_matrix_for_pair(x_raw[i][0], x_raw[i][1])

        tt = np.zeros((xx.shape[0],1), dtype=np.float)
        t_array = num_to_bin_array(t_raw[i])
        tt[:t_array.size,0] = t_array

        x.append(xx)
        t.append(tt)

    return x, t

def train_rnn_binary_add():
    print ''
    print 'Training RNN for binary add'
    print ''
    x_train, t_train = generate_binary_add_data(50, int_max=100)
    x_val, t_val = generate_binary_add_data(20, int_max=200)

    in_dim = x_train[0].shape[1]
    out_dim = t_train[0].shape[1]

    hid_dim = 20

    net = nn.NeuralNet(hid_dim, out_dim)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR)
    net.set_loss(ls.LOSS_NAME_SQUARED)

    rnn_net = rnn.RnnHybridNetwork(rnn.RNN(in_dim, hid_dim, nonlin_type=ly.NONLIN_NAME_TANH), net)

    print rnn_net

    rnn_learner = rnn.SequenceLearner(rnn_net)
    rnn_learner.load_data(x_train, t_train, x_val=x_val, t_val=t_val)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0.5, iprint=10, max_iters=200, max_grad_norm=10)
    # rnn_learner.train_gradient_descent(learn_rate=1e-3, momentum=0.9, iprint=10, max_iters=200)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0, iprint=10, max_iters=200, adagrad_start_iter=10)
    rnn_learner.train_sgd(minibatch_size=1, learn_rate=1e-1, momentum=0.9, iprint=100, adagrad_start_iter=1, max_iters=2000, max_grad_norm=1)

    return rnn_net

def compute_add(net, a, b):
    return bin_array_to_num(net.forward_prop(get_binary_matrix_for_pair(a, b)).asarray().round().astype(np.int).ravel())

def revert_sequence(x):
    new_x = np.empty(len(x), dtype=np.object)
    for i in xrange(len(x)):
        new_x[i] = x[i][::-1]
    return new_x

def train_rnn_ae():
    print ''
    print 'Training RNN autoencoder'
    print ''
    x_train, _ = generate_binary_add_data(50, int_max=64)
    x_val, _ = generate_binary_add_data(20, int_max=64)

    in_dim = x_train[0].shape[1]
    out_dim = in_dim

    hid_dim = 20

    net = nn.NeuralNet(hid_dim, out_dim)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR)
    net.set_loss(ls.LOSS_NAME_SQUARED)

    dec = rnn.RnnHybridNetwork(rnn.RNN(out_dim=hid_dim, nonlin_type=ly.NONLIN_NAME_TANH), net)
    enc = rnn.RNN(in_dim=in_dim, out_dim=hid_dim, nonlin_type=ly.NONLIN_NAME_TANH)

    ae = rnn.RnnAutoEncoder(encoder=enc, decoder=dec)

    print ae

    rnn_learner = rnn.SequenceLearner(ae)
    # rnn_learner.load_data(x_train, revert_sequence(x_train), x_val=x_val, t_val=revert_sequence(x_val))
    rnn_learner.load_data(x_train, x_train, x_val=x_val, t_val=x_val)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0.5, iprint=10, max_iters=200, max_grad_norm=10)
    # rnn_learner.train_gradient_descent(learn_rate=1e-3, momentum=0.9, iprint=10, max_iters=200)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0, iprint=10, max_iters=200, adagrad_start_iter=10)
    rnn_learner.train_sgd(minibatch_size=1, learn_rate=1e-1, momentum=0.9, iprint=100, adagrad_start_iter=1, max_iters=10000, max_grad_norm=1)

    return ae

def train_rnn_on_nn_ae():
    print ''
    print 'Training RNN autoencoder'
    print ''
    x_train, _ = generate_binary_add_data(50, int_max=64)
    x_val, _ = generate_binary_add_data(20, int_max=64)

    in_dim = x_train[0].shape[1]
    out_dim = in_dim

    hid_dim = 10
    out_hid_dim = 5
    in_hid_dim = 5

    net = nn.NeuralNet(hid_dim, out_dim)
    net.add_layer(out_hid_dim, nonlin_type=ly.NONLIN_NAME_RELU)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR)
    net.set_loss(ls.LOSS_NAME_SQUARED)

    dec = rnn.RnnHybridNetwork(rnn.RNN(out_dim=hid_dim, nonlin_type=ly.NONLIN_NAME_RELU), net)

    enc_net = nn.NeuralNet(in_dim, in_hid_dim)
    enc_net.add_layer(0, nonlin_type=ly.NONLIN_NAME_RELU)

    enc = rnn.RnnOnNeuralNet(enc_net, rnn.RNN(in_dim=in_hid_dim, out_dim=hid_dim, nonlin_type=ly.NONLIN_NAME_RELU))

    ae = rnn.RnnAutoEncoder(encoder=enc, decoder=dec)

    print ae

    rnn_learner = rnn.SequenceLearner(ae)
    # rnn_learner.load_data(x_train, revert_sequence(x_train), x_val=x_val, t_val=revert_sequence(x_val))
    rnn_learner.load_data(x_train, x_train, x_val=x_val, t_val=x_val)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0.5, iprint=10, max_iters=200, max_grad_norm=10)
    # rnn_learner.train_gradient_descent(learn_rate=1e-3, momentum=0.9, iprint=10, max_iters=200)
    # rnn_learner.train_gradient_descent(learn_rate=1e-2, momentum=0, iprint=10, max_iters=200, adagrad_start_iter=10)
    rnn_learner.train_sgd(minibatch_size=1, learn_rate=1e-1, momentum=0.9, iprint=100, adagrad_start_iter=1, max_iters=10000, max_grad_norm=1)

    return ae

if __name__ == '__main__':
    train_rnn_binary_add()
    train_rnn_ae()
