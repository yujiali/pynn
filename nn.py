"""
A python neural network package based on gnumpy.

Yujia Li, 09/2014
"""

import layer as ly
import gnumpy as gnp

class NetworkConstructionError(Exception):
    pass

class BaseNeuralNet(object):
    """
    Feed-forward neural network base class, each layer is fully connected.
    """
    def __init__(self):
        pass

    def forward_prop(self, X, compute_loss=False):
        """
        Do a forward propagation, which maps input matrix X (n_cases, n_dims)
        to an output matrix Y (n_cases, n_out_dims).

        compute_loss - compute all the losses if set.
        """
        raise NotImplementedError()

    def backward_prop(self, grad):
        """
        Given the gradients for the output layer, back propagate through the
        network and compute all the gradients.
        """
        raise NotImplementedError()

    def get_param_vec(self):
        """
        Get a vector representation of all parameters in the network.
        """
        raise NotImplementedError()

    def set_param_from_vec(self, v):
        """
        Set the parameters of the network from a complete vector representation.
        """
        raise NotImplementedError()

    def get_grad_vec(self):
        """
        Get a vector representation of all gradients for parameters in the network.
        """
        raise NotImplementedError()

    def save_model_to_binary(self):
        """
        Return a binary representation of the network.
        """
        raise NotImplementedError()

    def load_model_from_stream(self, f):
        """
        Load model from binary stream, f can be an open file.
        """
        raise NotImplementedError()

    def save_model_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            f.write(self.save_model_to_binary())

    def load_model_from_file(self, file_name):
        with open(file_name, 'rb') as f:
            self.load_model_from_stream(f)

class NeuralNet(BaseNeuralNet):
    """
    A simple one input one output layer neural net, loss is only (possibly) 
    added at the output layer.
    """
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.layer_params = []
        self.loss = None

        self.output_layer_added = False

    def add_layer(self, out_dim=0, nonlin_type=None, dropout=0, init_scale=1e-1,
            params=None):
        """
        By default, nonlinearity is linear.

        Return the newly added layer.
        """
        if self.output_layer_added:
            raise NetworkConstructionError(
                    'Trying to add more layers beyond output layer.')

        if len(self.layers) == 0:
            in_dim = self.in_dim
        else:
            in_dim = self.layers[-1].out_dim

        if out_dim == 0:
            out_dim = self.out_dim
            self.output_layer_added = True

        self.layers.append(ly.Layer(in_dim, out_dim, nonlin_type, dropout,
            init_scale, params))

        if params == None:
            self.layer_params.append(self.layers[-1].params)

        return self.layers[-1]

    def set_loss(self, loss):
        self.loss = loss
        self.layers[-1].set_loss(loss)

    def load_target(self, target, *args, **kwargs):
        self.loss.load_target(target, *args, **kwargs)

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        """
        Compute forward prop, return the output of the network.
        """
        if isinstance(X, gnp.garray):
            x_input = X
        else:
            x_input = gnp.garray(X)

        for i in range(len(self.layers)):
            x_input = self.layers[i].forward_prop(x_input, 
                    add_noise=add_noise, compute_loss=compute_loss)

        return x_input

    def get_loss(self):
        """
        Return the loss computed in a previous forward propagation.
        """
        return self.loss.get_most_recent_loss() if self.loss is not None else 0

