"""
Definitions of layers and nonlinearities in a neural network.

Yujia Li, 09/2014
"""

import gnumpy as gnp
import numpy as np
import const
import struct
import loss as ls

class LayerParams(object):
    """
    Parameters of a layer. Weight matrix shape is in_dim * out_dim.
    """
    _param_count = 0

    def __init__(self, in_dim=1, out_dim=1, init_scale=1e-1, dropout=0,
            in_stream=None):
        if in_stream is not None:
            self.load_from_stream(in_stream)
            return

        self.W = gnp.randn(in_dim, out_dim) * init_scale
        self.b = gnp.zeros(out_dim)

        self.W_grad = self.W * 0
        self.b_grad = self.b * 0

        self.param_size = self.W.size + self.b.size
        self.dropout = dropout

        # get an ID for this param variable.
        self._param_id = LayerParams._param_count
        LayerParams._param_count += 1

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

    def set_noiseless_param_from_vec(self, v):
        self.set_param_from_vec(v * (1 - self.dropout))

    def save_to_binary(self):
        s = struct.pack('iiif', self._param_id, self.W.shape[0],
                self.W.shape[1], self.dropout)
        s += self.W.asarray().astype(np.float32).tostring()
        s += self.b.asarray().astype(np.float32).tostring()
        return s

    def load_from_stream(self, f):
        self._param_id, self.in_dim, self.out_dim, self.dropout = \
                struct.unpack('iiif', f.read(4*4))
        self.W = gnp.garray(np.fromstring(f.read(self.in_dim * self.out_dim * 4), 
            dtype=np.float32).reshape(self.in_dim, self.out_dim))
        self.b = gnp.garray(np.fromstring(f.read(self.out_dim * 4), dtype=np.float32))
        
        self.W_grad = self.W * 0
        self.b_grad = self.b * 0

        self.param_size = self.W.size + self.b.size

class Layer(object):
    """
    One layer in a neural network.
    """
    def __init__(self, in_dim=1, out_dim=1, nonlin_type=None, dropout=0,
            init_scale=1e-1, params=None, loss=None):
        if nonlin_type is None:
            nonlin_type = NONLIN_NAME_LINEAR
        nonlin = get_nonlin_from_type_name(nonlin_type)
        self.build_layer(in_dim, out_dim, nonlin, dropout, init_scale, loss, params)

    def build_layer(self, in_dim, out_dim, nonlin, dropout=0, 
            init_scale=1e-1, loss=None, params=None):
        self.nonlin = nonlin
        self.set_params(params if params is not None else \
                LayerParams(in_dim, out_dim, init_scale, dropout))
        self.loss = loss
        self.loss_value = 0
        self.noise_added = False
        self.loss_computed = False

    def set_params(self, params):
        self.params = params
        self.in_dim, self.out_dim = params.W.shape
        self._param_id = params._param_id

    def set_loss(self, loss):
        self.loss = loss

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        """
        Compute the forward propagation step that maps the input data matrix X
        into the output. Loss and loss gradient will be computed when
        compute_loss set to True. Note that the loss is applied on nonlinearity
        activation, rather than the final output.
        """
        if self.params.dropout > 0 and add_noise:
            self.dropout_mask = gnp.rand(X.shape[0], X.shape[1]) > self.params.dropout
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
        if self.params.dropout > 0 and self.noise_added:
            d_input *= self.dropout_mask

        self.params.add_gradient(self.inputs.T.dot(d_act), d_act.sum(axis=0))
        return d_input

    def save_to_binary(self):
        return struct.pack('ii', self.in_dim, self.out_dim) \
                + struct.pack('i', self.params._param_id) \
                + struct.pack('i', self.nonlin.get_id()) \
                + (struct.pack('i', self.loss.get_id() \
                if self.loss is not None else ls._LOSS_ID_NONE)) \
                + (struct.pack('f', self.loss.weight \
                if self.loss is not None else 0))

    def load_from_stream(self, f):
        in_dim, out_dim, _param_id, nonlin_id, loss_id, loss_weight = \
                struct.unpack('iiiiif', f.read(6*4))
        nonlin = get_nonlin_from_type_id(nonlin_id)
        loss = ls.get_loss_from_type_id(loss_id)
        if loss is not None:
            loss.set_weight(loss_weight)

        self.build_layer(in_dim, out_dim, nonlin, 
                init_scale=const.DEFAULT_PARAM_INIT_SCALE, loss=loss)
        self._param_id = _param_id

    def __repr__(self):
        return '%s %d x %d, dropout %g' % (self.nonlin.get_name(), 
                self.in_dim, self.out_dim, self.params.dropout)

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

    def get_id(self):
        raise NotImplementedError()

class NonlinManager(object):
    """
    Maintains a set of nonlinearities.
    """
    def __init__(self):
        self.name_to_nonlin = {}
        self.id_to_nonlin = {}

    def register_nonlin(self, nonlin):
        nonlin_name = nonlin.get_name()
        nonlin_id = nonlin.get_id()
        self.name_to_nonlin[nonlin_name] = nonlin
        self.id_to_nonlin[nonlin_id] = nonlin
        globals()['NONLIN_NAME_' + nonlin_name.upper()] = nonlin_name
        globals()['_NONLIN_ID_' + nonlin_name.upper()] = nonlin_id

    def get_nonlin_list(self):
        return self.name_to_nonlin.values()

    def get_nonlin_instance(self, nonlin_type):
        return self.name_to_nonlin[nonlin_type]

    def get_nonlin_instance_from_id(self, nonlin_id):
        return self.id_to_nonlin[nonlin_id]

_nonlin_manager = NonlinManager()

def register_nonlin(nonlin):
    _nonlin_manager.register_nonlin(nonlin)

def get_nonlin_from_type_name(nonlin_type):
    return _nonlin_manager.get_nonlin_instance(nonlin_type)

def get_nonlin_from_type_id(nonlin_id):
    return _nonlin_manager.get_nonlin_instance_from_id(nonlin_id)

# Definitions for all nonlinearities start here.

class LinearNonlin(Nonlinearity):
    def forward_prop(self, x):
        return x

    def backward_prop(self, x, z):
        return gnp.ones(x.shape)
        # return gnp.garray(1)

    def get_name(self):
        return 'linear'

    def get_id(self):
        return 0

register_nonlin(LinearNonlin())

class SigmoidNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.logistic(x)

    def backward_prop(self, x, z):
        return z * (1 - z)

    def get_name(self):
        return 'sigmoid'

    def get_id(self):
        return 1

register_nonlin(SigmoidNonlin())

class TanhNonlin(Nonlinearity):
    def forward_prop(self, x):
        return gnp.tanh(x)

    def backward_prop(self, x, z):
        return 1 - z**2

    def get_name(self):
        return 'tanh'

    def get_id(self):
        return 2

register_nonlin(TanhNonlin())

class ReluNonlin(Nonlinearity):
    def forward_prop(self, x):
        return x * (x > 0)

    def backward_prop(self, x, z):
        return x > 0

    def get_name(self):
        return 'relu'

    def get_id(self):
        return 3

register_nonlin(ReluNonlin())

NONLIN_LIST = _nonlin_manager.get_nonlin_list()

