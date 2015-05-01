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
            in_stream=None, init_bias=0):
        if in_stream is not None:
            self.load_from_stream(in_stream)
            return

        self.W = gnp.randn(in_dim, out_dim) * init_scale
        self.b = gnp.ones(out_dim) * init_bias

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
        self.set_param_from_vec(v / (1 - self.dropout))

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
            sparsity=0, sparsity_weight=0, init_scale=1e-1, params=None,
            loss=None, loss_after_nonlin=False, init_bias=0):
        if nonlin_type is None:
            nonlin_type = NONLIN_NAME_LINEAR
        nonlin = get_nonlin_from_type_name(nonlin_type)
        self.build_layer(in_dim, out_dim, nonlin, dropout=dropout, 
                sparsity=sparsity, sparsity_weight=sparsity_weight,
                init_scale=init_scale, loss=loss, params=params,
                loss_after_nonlin=loss_after_nonlin)

    def build_layer(self, in_dim, out_dim, nonlin, dropout=0, sparsity=0, sparsity_weight=0,
            init_scale=1e-1, loss=None, params=None, loss_after_nonlin=False, init_bias=0):
        self.nonlin = nonlin
        self.set_params(params if params is not None else \
                LayerParams(in_dim, out_dim, init_scale, dropout, init_bias=init_bias))

        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight
        if self.sparsity_weight > 0:
            self._sparsity_current = gnp.ones(out_dim) * sparsity
            self._sparsity_smoothing = 0.9
            self._sparsity_objective = 0

        self.loss = loss
        self.loss_value = 0
        self.noise_added = False
        self.loss_computed = False
        self.loss_after_nonlin = loss_after_nonlin

    def set_params(self, params):
        self.params = params
        self.in_dim, self.out_dim = params.W.shape
        self._param_id = params._param_id

    def set_loss(self, loss, loss_after_nonlin=False):
        self.loss = loss
        self.loss_after_nonlin = loss_after_nonlin

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

        if self.sparsity_weight > 0:
            self._sparsity_current = self._sparsity_smoothing * self.output.mean(axis=0) \
                    + (1 - self._sparsity_smoothing) * self._sparsity_current
            self._sparsity_objective = (- self.sparsity * gnp.log(self._sparsity_current + 1e-20) \
                    - (1 - self.sparsity) * gnp.log(1 - self._sparsity_current + 1e-20)).sum() * self.sparsity_weight

        if compute_loss and self.loss is not None:
            if self.loss_after_nonlin:
                self.loss_value, self.loss_grad = self.loss.compute_loss_and_grad(
                        self.output, compute_grad=True)
            else:
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
        if grad is None and not self.loss_after_nonlin and self.sparsity_weight == 0:
            d_act = gnp.zeros(self.output.shape)
        else:   # some gradient will pass through nonlinearity
            if grad is None and self.loss_after_nonlin:
                grad = self.loss_grad
            elif self.loss_after_nonlin:
                grad += self.loss_grad

            if self.sparsity_weight > 0:
                sparsity_grad = self.sparsity_weight * self._sparsity_smoothing \
                        / self.output.shape[0] * (self._sparsity_current - self.sparsity) \
                        / (self._sparsity_current * (1 - self._sparsity_current))
                if grad is None:
                    grad = sparsity_grad
                else:
                    grad += sparsity_grad

            d_act = self.nonlin.backward_prop(self.activation, self.output) * grad 

        if self.loss_computed and not self.loss_after_nonlin:
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
                if self.loss is not None else 0)) \
                + (struct.pack('i', 1 if self.loss_after_nonlin else 0)) \
                + struct.pack('f', self.sparsity) \
                + struct.pack('f', self.sparsity_weight)

    def load_from_stream(self, f):
        in_dim, out_dim, _param_id, nonlin_id, loss_id, loss_weight, loss_after, \
                sparsity, sparsity_weight = struct.unpack('iiiiififf', f.read(9*4))
        nonlin = get_nonlin_from_type_id(nonlin_id)
        loss = ls.get_loss_from_type_id(loss_id)

        if loss is not None:
            loss.set_weight(loss_weight)

        self.build_layer(in_dim, out_dim, nonlin,
                sparsity=sparsity, sparsity_weight=sparsity_weight,
                init_scale=const.DEFAULT_PARAM_INIT_SCALE, loss=loss,
                loss_after_nonlin=(loss_after==1))
        self._param_id = _param_id

    def __repr__(self):
        return '%s %d x %d, dropout %g' % (self.nonlin.get_name(), 
                self.in_dim, self.out_dim, self.params.dropout) \
                + (', loss after' if self.loss_after_nonlin else '') \
                + (', sparsity %g, sparsity_weight %g' % (self.sparsity, self.sparsity_weight) if self.sparsity_weight > 0 else '')

    def get_status_info(self):
        return '' if self.sparsity_weight == 0 else 'layer %dx%d sparsity=%g' % (
                self.in_dim, self.out_dim, self._sparsity_current.mean())

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

    def invert_output(self, z):
        """
        z: computed output from forward_prop

        Return a matrix same size as z, which is the inferred input that could
        have been used as the input to forward_prop to get z.

        Returns z by default if this function is not implemented by subclasses.
        """
        return z

    def output_range(self):
        """
        Return a tuple (min, max) of possible outputs.  If there's no lower
        bound on the min, then min=None, and if no upper bound on max then 
        max=None.
        """
        return None, None

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

    def invert_output(self, z):
        return z

    def output_range(self):
        return None, None

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

    def invert_output(self, z):
        return gnp.log(z / (1 - z))

    def output_range(self):
        return 0, 1

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

    def invert_output(self, z):
        return 0.5 * gnp.log((1+z) / (1-z))

    def output_range(self):
        return -1, 1

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

    def output_range(self):
        return 0, None

    def get_name(self):
        return 'relu'

    def get_id(self):
        return 3

register_nonlin(ReluNonlin())

NONLIN_LIST = _nonlin_manager.get_nonlin_list()

