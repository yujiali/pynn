"""
Definitions of layers and nonlinearities in a neural network.

Yujia Li, 09/2014
"""

import gnumpy as gnp
import numpy as np
import const

class LayerParams(object):
    """
    Parameters of a layer. Weight matrix shape is in_dim * out_dim.
    """
    def __init__(self, in_dim, out_dim, init_scale=1e-1, dropout=0):
        self.W = gnp.randn(in_dim, out_dim) * init_scale
        self.b = gnp.zeros(out_dim)

        self.W_grad = self.W * 0
        self.b_grad = self.b * 0

        self.param_size = self.W.size + self.b.size
        self.dropout = dropout

    def clear_gradient(self):
        self.W_grad[:] = 0
        self.b_grad[:] = 0

    def add_gradient(self, dW, db):
        self.W_grad += dW
        self.b_grad += db

    def set_gradient(self, dW, db):
        self.W_grad = dW
        self.b_grad = db

    def get_param_vec(self):
        return np.r_[self.W.asarray().ravel(), self.b.asarray().ravel()]

    def get_noiseless_param_vec(self):
        v = self.get_param_vec() 
        return v * (1 - self.dropout) if self.dropout > 0 else v

    def get_grad_vec(self):
        return np.r_[self.W_grad.asarray().ravel(), self.b_grad.asarray().ravel()]

    def set_param_from_vec(self, v):
        self.W[:] = v[:self.W.size].reshape(self.W.shape)
        self.b[:] = v[self.W.size:].reshape(self.b.shape)

class Layer(object):
    """
    One layer in a neural network.
    """
    def __init__(self, in_dim, out_dim, nonlin_type=None, dropout=0,
            init_scale=1e-1, params=None):
        self.in_dim = in_dim
        self.out_dim = out_dim

        if nonlin_type is None:
            nonlin_type = NONLIN_NAME_LINEAR
        self.nonlin = get_nonlin_from_type_name(nonlin_type)

        self.dropout = dropout

        if params is not None:
            self.params = params
            self.dropout = params.dropout
        else:
            self.params = LayerParams(in_dim, out_dim, init_scale, dropout)

    def forward_prop(self, X, add_noise=False):
        """
        Compute the forward propagation step that maps the input data matrix X
        into the output.
        """
        if self.dropout > 0 and add_noise:
            self.dropout_mask = gnp.rand(X.shape[0], X.shape[1]) > self.dropout
            self.inputs = X * self.dropout_mask
        else:
            self.inputs = X
        self.noise_added = add_noise

        self.activation = self.inputs.dot(self.params.W) + self.params.b
        self.output = self.nonlin.forward_prop(self.activation)
        return self.output

    def backward_prop(self, grad):
        """
        Compute the backward propagation step, with output gradient as input.

        Compute gradients for the input and update the gradient for the weights.
        """
        d_act = self.nonlin.backward_prop(self.activation, self.output) * grad
        d_input = d_act.dot(self.params.W.T)
        if self.dropout > 0 and self.noise_added:
            d_input *= self.dropout_mask
        self.params.add_gradient(self.inputs.T.dot(d_act), d_act.sum(axis=0))
        return d_input

    def __repr__(self):
        return '[ %s %d x %d ]' % (self.nonlin.get_name(), self.in_dim, self.out_dim)

class Nonlinearity(object):
    """
    Base class for the nonlinearities. These nonlinearities are applied in an
    element-wise fashion to all input dimensions.
    """
    def __init__(self):
        pass

    def forward_prop(self, x):
        """
        x: input
        """
        raise NotImplementedError()

    def backward_prop(self, x, z):
        """
        z: computed output from forward_prop
        """
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

_nonlin_dict = {}

# Definitions for all nonlinearities start here.

class LinearNonlin(Nonlinearity):
    def forward_prop(self, x):
        return X

    def backward_prop(self, x, z):
        return gnp.ones(x.shape, dtype=x.dtype)

    def get_name(self):
        return 'linear'

_linear_nonlin = LinearNonlin()
NONLIN_NAME_LINEAR = _linear_nonlin.get_name()
_nonlin_dict[NONLIN_NAME_LINEAR] = _linear_nonlin

class SigmoidNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.logistic(x)

    def backward_prop(self, x, z):
        return z * (1 - z)

    def get_name(self):
        return 'sigmoid'

_sigmoid_nonlin = SigmoidNonlin()
NONLIN_NAME_SIGMOID = _sigmoid_nonlin.get_name()
_nonlin_dict[NONLIN_NAME_SIGMOID] = _sigmoid_nonlin

class TanhNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.tanh(x)

    def backward_prop(self, x, z):
        return 1 - z**2

    def get_name(self):
        return 'tanh'

_tanh_nonlin = TanhNonlin()
NONLIN_NAME_TANH = _tanh_nonlin.get_name()
_nonlin_dict[NONLIN_NAME_TANH] = _tanh_nonlin

class ReluNonlin(Nonlinearity):
    def forward_prop(self, x):
        return x * (x > 0)

    def backward_prop(self, x, z):
        return x > 0

    def get_name(self):
        return 'relu'

_relu_nonlin = ReluNonlin()
NONLIN_NAME_RELU = _relu_nonlin.get_name()
_nonlin_dict[NONLIN_NAME_RELU] = _relu_nonlin

def get_nonlin_from_type_name(nonlin_type):
    return _nonlin_dict[nonlin_type]

