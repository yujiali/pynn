"""
A python neural network package based on gnumpy.

Yujia Li, 09/2014
"""

import layer as ly

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
    A simple one input one output layer neural net.
    """
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.layer_params = []
        self.loss = None

        self.output_layer_added = False

    def add_layer(self, out_dim=0, nonlin_type=None, dropout=0, init_scale=1e-1,
            params=None, loss_type=None):
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

    def forward_prop(self, X, add_noise=False):
        pass

