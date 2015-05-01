"""
A python neural network package based on gnumpy.

Yujia Li, 09/2014

TODO:
- right now YNeuralNet I/O only supports NeuralNet as the type for component
  nets (network construction and forward/backward prop works for other types
  of component nets just fine). Ideally this should be extended to 
  StackedNeuralNet and other types as well.
"""

import gnumpy as gnp
import numpy as np
import layer as ly
import loss as ls
import struct

class NetworkConstructionError(Exception):
    pass

class NetworkCompositionError(Exception):
    pass

class TargetLoadingError(Exception):
    pass

class BaseNeuralNet(object):
    """
    Feed-forward neural network base class, each layer is fully connected.
    """
    def __init__(self):
        pass

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        """
        Do a forward propagation, which maps input matrix X (n_cases, n_dims)
        to an output matrix Y (n_cases, n_out_dims).

        add_noise - add noise if set.
        compute_loss - compute all the losses if set.
        """
        raise NotImplementedError()

    def load_target(self, *args, **kwargs):
        """
        Load targets used in the losses.
        """
        raise NotImplementedError()

    def get_loss(self):
        """
        Return the loss computed in a previous forward propagation.
        """
        raise NotImplementedError()

    def backward_prop(self, grad=None):
        """
        Given the gradients for the output layer, back propagate through the
        network and compute all the gradients.
        """
        raise NotImplementedError()

    def clear_gradient(self):
        """
        Reset all parameter gradients to 0.
        """
        raise NotImplementedError()

    def get_param_vec(self):
        """
        Get a vector representation of all parameters in the network.
        """
        raise NotImplementedError()

    def get_noiseless_param_vec(self):
        """
        Get an approximate vector representation of all parameters in the
        network, that corresponds to the noiseless case when using dropout in
        training.
        """
        return self.get_param_vec()

    def _set_param_from_vec(self, v, is_noiseless=False):
        """
        is_noiseless=True -> set_noiseless_param_from_vec,
        is_noiseless=False -> set_param_from_vec
        """
        raise NotImplementedError()

    def set_param_from_vec(self, v):
        """
        Set the parameters of the network from a complete vector representation.
        """
        self._set_param_from_vec(v, is_noiseless=False)

    def set_noiseless_param_from_vec(self, v):
        """
        Set the parameters of the network from a complete vector representation,
        but properly scale it to be used in noiseless setting.
        """
        self._set_param_from_vec(v, is_noiseless=True)

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

    def get_type_code(self):
        """
        A type code used in model I/O to distinguish among different models.

        This should return a 32-bit integer.
        """
        raise NotImplementedError()

    def check_type_code(self, type_code):
        """
        Check if the type code matches the model itself.
        """
        if type_code == self.get_type_code():
            return
        else:
            raise Exception('Type code mismatch!')

    def _update_param_size(self):
        """
        Update parameter size. After a call to this function the param_size
        attribute will be set properly.
        """
        raise NotImplementedError()

    def get_status_info(self):
        """
        Return a string that represents some internal states of the network,
        can be used for debugging the training process or monitoring the state
        of the network.
        """
        return ''

class NeuralNet(BaseNeuralNet):
    """
    A simple one input one output layer neural net, loss is only (possibly) 
    added at the output layer.
    """
    def __init__(self, in_dim=None, out_dim=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.layer_params = []
        self.param_size = 0
        self.loss = None

        self.output_layer_added = False

    def add_layer(self, out_dim=0, nonlin_type=None, dropout=0, sparsity=0, 
            sparsity_weight=0, init_scale=1e-1, params=None, init_bias=0):
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

        if params is not None:
            if in_dim != params.W.shape[0]:
                raise NetworkConstructionError(
                        'Loading shared parameter failure: size mismatch.')
            else:
                out_dim = params.W.shape[1]

        if out_dim == 0:
            out_dim = self.out_dim
            self.output_layer_added = True

        self.layers.append(ly.Layer(in_dim, out_dim, nonlin_type, dropout,
            sparsity, sparsity_weight, init_scale, params, init_bias=init_bias))

        if params is None:
            self.layer_params.append(self.layers[-1].params)

        self._update_param_size()

        return self.layers[-1]

    def _update_param_size(self):
        self.param_size = sum([p.param_size for p in self.layer_params])

    def set_loss(self, loss_type, loss_weight=1, loss_after_nonlin=False, **kwargs):
        """
        loss_type is the name of the loss.
        """
        self.loss = ls.get_loss_from_type_name(loss_type, **kwargs)
        self.loss.set_weight(loss_weight)
        self.layers[-1].set_loss(self.loss, loss_after_nonlin=loss_after_nonlin)

    def load_target(self, target, *args, **kwargs):
        if self.loss is not None and target is not None:
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

    def clear_gradient(self):
        """
        Reset all parameter gradients to zero.
        """
        for p in self.layer_params:
            p.clear_gradient()

    def backward_prop(self, grad=None):
        """
        Compute the backward prop, return the input gradient.
        """
        for i in range(len(self.layers))[::-1]:
            grad = self.layers[i].backward_prop(grad)
        return grad

    def get_param_vec(self):
        return np.concatenate([self.layer_params[i].get_param_vec() \
                for i in range(len(self.layer_params))])

    def get_noiseless_param_vec(self):
        return np.concatenate([self.layer_params[i].get_noiseless_param_vec() \
                for i in range(len(self.layer_params))])

    def _set_param_from_vec(self, v, is_noiseless=False):
        i_start = 0
        for i in range(len(self.layer_params)):
            p = self.layer_params[i]
            if is_noiseless:
                p.set_noiseless_param_from_vec(v[i_start:i_start+p.param_size])
            else:
                p.set_param_from_vec(v[i_start:i_start+p.param_size])
            i_start += p.param_size

    def get_grad_vec(self):
        return np.concatenate([self.layer_params[i].get_grad_vec() \
                for i in range(len(self.layer_params))])

    #def noiseless_mode_setup(self):
    #    self.set_param_from_vec(self.get_noiseless_param_vec())
    #    for p in self.layer_params:
    #        p.dropout = 0

    def __repr__(self):
        return ' | '.join([str(self.layers[i]) for i in range(len(self.layers))]) \
                + ' | ' + (str(self.loss) if self.loss is not None else 'No Loss')

    def get_type_code(self):
        return 0

    def save_model_to_binary(self):
        # network structure first
        s = struct.pack('i', self.get_type_code())
        s += struct.pack('i', len(self.layers))
        s += ''.join([self.layers[i].save_to_binary() \
                for i in range(len(self.layers))])

        # network parameters
        s += struct.pack('i', len(self.layer_params))
        s += ''.join([self.layer_params[i].save_to_binary() \
                for i in range(len(self.layer_params))])
        return s

    def load_model_from_stream(self, f):
        self.layers = []
        self.layer_params = []

        type_code = struct.unpack('i', f.read(4))[0]
        self.check_type_code(type_code)

        n_layers = struct.unpack('i', f.read(4))[0]
        for i in range(n_layers):
            layer = ly.Layer()
            layer.load_from_stream(f)
            self.layers.append(layer)

        n_params = struct.unpack('i', f.read(4))[0]
        for i in range(n_params):
            p = ly.LayerParams(in_stream=f)
            self.layer_params.append(p)

            for layer in self.layers:
                if layer._param_id == p._param_id:
                    layer.set_params(p)

        self.in_dim = self.layers[0].in_dim
        self.out_dim = self.layers[-1].out_dim
        self.loss = self.layers[-1].loss
        
        self.output_layer_added = False
        self._update_param_size()

    def get_status_info(self):
        return ', '.join([s for s in [layer.get_status_info() for layer in self.layers] if len(s) > 0])

class CompositionalNeuralNet(BaseNeuralNet):
    """
    A base class for all meta neural nets that are formed by combining multiple
    different nets.
    """
    def __init__(self, *neural_nets):
        self.neural_nets = neural_nets
        self._update_param_size()

    def _update_param_size(self):
        self.param_size = sum([net.param_size for net in self.neural_nets])

    def get_loss(self):
        return sum([net.get_loss() for net in self.neural_nets])

    def clear_gradient(self):
        for net in self.neural_nets:
            net.clear_gradient()

    def get_param_vec(self):
        return np.concatenate([self.neural_nets[i].get_param_vec() \
                for i in range(len(self.neural_nets))])

    def get_noiseless_param_vec(self):
        return np.concatenate([self.neural_nets[i].get_noiseless_param_vec() \
                for i in range(len(self.neural_nets))])

    def _set_param_from_vec(self, v, is_noiseless=False):
        i_start = 0
        for i in range(len(self.neural_nets)):
            net = self.neural_nets[i]
            if is_noiseless:
                net.set_noiseless_param_from_vec(v[i_start:i_start+net.param_size])
            else:
                net.set_param_from_vec(v[i_start:i_start + net.param_size])
            i_start += net.param_size

    def get_grad_vec(self):
        return np.concatenate([self.neural_nets[i].get_grad_vec() \
                for i in range(len(self.neural_nets))])

    def save_model_to_binary(self):
        return struct.pack('i', len(self.neural_nets)) \
                + ''.join([self.neural_nets[i].save_model_to_binary() \
                for i in range(len(self.neural_nets))])

    def load_model_from_stream(self, f):
        n_nets = struct.unpack('i', f.read(4))[0]
        self.neural_nets = []
        for i in range(n_nets):
            net = NeuralNet(0, 0)
            net.load_model_from_stream(f)
            self.neural_nets.append(net)

    def get_status_info(self):
        return ', '.join([s for s in [net.get_status_info() for net in self.neural_nets] if len(s) > 0])

class StackedNeuralNet(CompositionalNeuralNet):
    """
    Create a new network by stacking a few smaller NeuralNets.
    """
    def __init__(self, *neural_nets):
        super(StackedNeuralNet, self).__init__(*neural_nets)

        if len(neural_nets) > 0:
            self.in_dim = neural_nets[0].in_dim
            self.out_dim = neural_nets[-1].out_dim

    def load_target(self, *args):
        # place holder case, where no target is loaded
        if len(args) == 1 and args[0] is None:
            return

        if len(args) == 1 and isinstance(args[0], list):
            targets = args[0]
        else:
            targets = args

        if len(targets) != len(self.neural_nets):
            raise NetworkCompositionError('Number of loss targets should be the' \
                    + ' same as number of stacked neural nets.')

        for i in range(len(targets)):
            self.neural_nets[i].load_target(targets[i])

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        x_input = X
        for i in range(len(self.neural_nets)):
            x_input = self.neural_nets[i].forward_prop(x_input, 
                    add_noise=add_noise, compute_loss=compute_loss)
        return x_input

    def backward_prop(self, grad=None):
        for i in range(len(self.neural_nets))[::-1]:
            grad = self.neural_nets[i].backward_prop(grad)
        return grad

    def load_model_from_stream(self, f):
        super(StackedNeuralNet, self).load_model_from_stream(f)

        self.in_dim = self.neural_nets[0].in_dim
        self.out_dim = self.neural_nets[-1].out_dim
        self._update_param_size()

    def __repr__(self):
        return '{ ' + ' }--{ '.join([str(self.neural_nets[i]) \
                for i in range(len(self.neural_nets))]) + ' }'

class YNeuralNet(CompositionalNeuralNet):
    """
    Create a new network of Y-shape
             +--> y
         (1) | (2)
        x -> h
             | (3)
             +--> z
    from (1) (2) and (3) three component networks.

    Note the Y-shape network does not have out_dim and output, as there are 
    two outputs.
    """
    def __init__(self, in_net=None, out_net1=None, out_net2=None):
        if (in_net is None) or (out_net1 is None) or (out_net2 is None):
            return

        super(YNeuralNet, self).__init__(in_net, out_net1, out_net2)
        self.in_dim = in_net.in_dim

        # for easy reference
        self.in_net = self.neural_nets[0]
        self.out_net1 = self.neural_nets[1]
        self.out_net2 = self.neural_nets[2]

    def load_target(self, *args):
        """
        args can be a single list, or three variables
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        elif len(args) != 3:
            raise TargetLoadingError('Target misspecified.')

        self.in_net.load_target(args[0])
        self.out_net1.load_target(args[1])
        self.out_net2.load_target(args[2])

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        h = self.in_net.forward_prop(X, add_noise=add_noise, compute_loss=compute_loss)
        self.out_net1.forward_prop(h, add_noise=add_noise, compute_loss=compute_loss)
        self.out_net2.forward_prop(h, add_noise=add_noise, compute_loss=compute_loss)

    def backward_prop(self):
        grad = self.out_net1.backward_prop()
        grad += self.out_net2.backward_prop()
        grad = self.in_net.backward_prop(grad)
        return grad

    def load_model_from_stream(self, f):
        super(YNeuralNet, self).load_model_from_stream(f)
        self.in_dim = self.neural_nets[0].in_dim
        self.in_net = self.neural_nets[0]
        self.out_net1 = self.neural_nets[1]
        self.out_net2 = self.neural_nets[2]
        self._update_param_size()

    def __repr__(self):
        s = '{ ' + str(self.in_net) + ' }'
        return len(s) * ' ' + '  +--{ ' + str(self.out_net1) + ' }\n' \
                + s + '--+\n' \
                + len(s) * ' ' + '  +--{ ' + str(self.out_net2) + ' }'

class AutoEncoder(CompositionalNeuralNet):
    """
    AutoEncoder network, with one encoder and one decoder.
    """
    def __init__(self, encoder=None, decoder=None):
        # place holder constructor when either of encoder/decoder is None
        if encoder is None or decoder is None:
            return
        super(AutoEncoder, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.in_dim = encoder.in_dim
        self.out_dim = decoder.out_dim

    def load_target(self, *args):
        pass

    def forward_prop(self, X, add_noise=False, compute_loss=False):
        """
        Equivalently this computes the reconstruction.
        """
        # input is the target
        if compute_loss:
            self.decoder.load_target(X)

        h = self.encoder.forward_prop(X, add_noise=add_noise,
                compute_loss=compute_loss)
        return self.decoder.forward_prop(h, add_noise=add_noise,
                compute_loss=compute_loss)

    def encode(self, X):
        return self.encoder.forward_prop(X, add_noise=False, compute_loss=False)

    def backward_prop(self):
        grad = self.decoder.backward_prop()
        return self.encoder.backward_prop(grad)

    def load_model_from_stream(self, f):
        super(AutoEncoder, self).load_model_from_stream(f)
        self.encoder = self.neural_nets[0]
        self.decoder = self.neural_nets[1]
        self.in_dim = self.encoder.in_dim
        self.out_dim = self.decoder.out_dim

    def __repr__(self):
        return 'Encoder { ' + str(self.encoder) + '} Decoder { ' + str(self.decoder) + ' }'


