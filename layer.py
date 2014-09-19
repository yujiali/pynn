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
            init_scale=1e-1, params=None, loss=None):
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

        self.loss = loss
        self.loss_value = 0

        self.noise_added = False
        self.loss_computed = False

    def set_loss(self, loss):
        self.loss = loss

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        """
        Compute the forward propagation step that maps the input data matrix X
        into the output. Loss and loss gradient will be computed when
        compute_loss set to True. Note that the loss is applied on nonlinearity
        activation, rather than the final output.
        """
        if self.dropout > 0 and add_noise:
            self.dropout_mask = gnp.rand(X.shape[0], X.shape[1]) > self.dropout
            self.inputs = X * self.dropout_mask
        else:
            self.inputs = X
        self.noise_added = add_noise

        self.activation = self.inputs.dot(self.params.W) + self.params.b
        self.output = self.nonlin.forward_prop(self.activation)

        if compute_loss and self.loss is not None:
            self.loss_value, self.loss_grad = self.loss.compute_loss_and_grad(
                    self.activation, compute_grad=True)
            self.loss_computed = True
        
        return self.output

    def backward_prop(self, grad=None):
        """
        Compute the backward propagation step, with output gradient as input.
        Compute gradients for the input and update the gradient for the weights.
        Note te loss gradients are added to the activation gradient, i.e. they 
        won't pass through the nonlinearity.
        """
        if grad is None:
            d_act = gnp.zeros(self.output.shape)
        else:
            d_act = self.nonlin.backward_prop(self.activation, self.output) * grad 

        if self.loss_computed:
            d_act += self.loss_grad

        d_input = d_act.dot(self.params.W.T)
        if self.dropout > 0 and self.noise_added:
            d_input *= self.dropout_mask
        self.params.add_gradient(self.inputs.T.dot(d_act), d_act.sum(axis=0))
        return d_input

    def __repr__(self):
        return '%s %d x %d' % (self.nonlin.get_name(), self.in_dim, self.out_dim)

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

        Return a matrix same size as x.
        """
        raise NotImplementedError()

    def backward_prop(self, x, z):
        """
        z: computed output from forward_prop

        Return a matrix same size as x and z
        """
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class NonlinManager(object):
    """
    Maintains a set of nonlinearities.
    """
    def __init__(self):
        self.nonlin_dict = {}

    def register_nonlin(self, nonlin):
        nonlin_name = nonlin.get_name()
        self.nonlin_dict[nonlin_name] = nonlin
        globals()['NONLIN_NAME_' + nonlin_name.upper()] = nonlin_name

    def get_nonlin_list(self):
        return self.nonlin_dict.values()

    def get_nonlin_instance(self, nonlin_type):
        return self.nonlin_dict[nonlin_type]

_nonlin_manager = NonlinManager()

def register_nonlin(nonlin):
    _nonlin_manager.register_nonlin(nonlin)

def get_nonlin_from_type_name(nonlin_type):
    return _nonlin_manager.get_nonlin_instance(nonlin_type)

# Definitions for all nonlinearities start here.

class LinearNonlin(Nonlinearity):
    def forward_prop(self, x):
        return x

    def backward_prop(self, x, z):
        return gnp.ones(x.shape)
        # return gnp.garray(1)

    def get_name(self):
        return 'linear'

register_nonlin(LinearNonlin())

class SigmoidNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.logistic(x)

    def backward_prop(self, x, z):
        return z * (1 - z)

    def get_name(self):
        return 'sigmoid'

register_nonlin(SigmoidNonlin())

class TanhNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.tanh(x)

    def backward_prop(self, x, z):
        return 1 - z**2

    def get_name(self):
        return 'tanh'

register_nonlin(TanhNonlin())

class ReluNonlin(Nonlinearity):
    def forward_prop(self, x):
        return x * (x > 0)

    def backward_prop(self, x, z):
        return x > 0

    def get_name(self):
        return 'relu'

register_nonlin(ReluNonlin())

NONLIN_LIST = _nonlin_manager.get_nonlin_list()

