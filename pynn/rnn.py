"""
Recurrent neural networks.

TODO: this is not a complete implementation.

Yujia Li, 05/2015
"""

import layer
import nn
import numpy as np
import gnumpy as gnp
import math
import struct

class RNN(nn.BaseNeuralNet):
    def __init__(self, in_dim=None, out_dim=None, nonlin_type=layer.NONLIN_NAME_TANH):
        if out_dim is None:
            return

        self.in_dim = in_dim
        self.has_input = in_dim is not None
        self.out_dim = out_dim
        self.nonlin = layer.get_nonlin_from_type_name(nonlin_type)

        self._init_params()
        self._update_param_size()

    def _init_params(self):
        if self.has_input:
            self.W_ih = gnp.randn(self.in_dim, self.out_dim) / math.sqrt(self.in_dim)
            self.dW_ih = self.W_ih * 0

        self.W_hh = gnp.eye(self.out_dim)
        self.b = gnp.zeros(self.out_dim)

        self.dW_hh = self.W_hh * 0
        self.db = self.b * 0

        self._update_param_size()

    def forward_prop(self, X=None, T=10, h_init=None, **kwargs):
        """
        options:
        - X can be None, when there's no input, then T must be specified
        - if X is not None, T will not be used
        - an extra h_init can be given to the forward prop to feed into the
          first hidden state activation.
        """
        if X is not None and self.has_input:
            X = gnp.as_garray(X)
            self.X = X

            T = X.shape[0]

            self.A = X.dot(self.W_ih) + self.b
        else:
            self.X = None
            self.A = self.b.tile((T,1))

        self.H = gnp.empty((T, self.out_dim))

        if h_init is not None:
            self.h_init = gnp.as_garray(h_init)
            self.A[0] += self.h_init.reshape(1,-1).dot(self.W_hh)
        else:
            self.h_init = None

        self.H[0] = self.nonlin.forward_prop(self.A[0])

        for t in range(1, T):
            self.A[t] += self.H[t-1].reshape(1,-1).dot(self.W_hh)
            self.H[t] = self.nonlin.forward_prop(self.A[t])

        return self.H

    def backward_prop(self, grad=None, grad_end=None):
        if grad is not None:
            T = grad.shape[0]
            assert T == self.H.shape[0]

            dH = grad.copy()
        else:
            T = self.H.shape[0]
            dH = gnp.zeros((T, self.H.shape[1]))

        if grad_end is not None:
            dH[-1] += gnp.as_garray(grad_end).ravel()

        dA = gnp.empty((dH.shape[0], dH.shape[1]))

        for t in range(1,T)[::-1]:
            dA[t] = self.nonlin.backward_prop(self.A[t], self.H[t]) * dH[t]
            dH[t-1] += self.W_hh.dot(dA[t].reshape(-1,1)).ravel()
        dA[0] = self.nonlin.backward_prop(self.A[0], self.H[0]) * dH[0]

        self.dW_hh += self.H[:-1].T.dot(dA[1:])

        if self.h_init is not None:
            self.dW_hh += self.h_init.reshape(-1,1).dot(dA[0].reshape(1,-1))

        self.db += dA.sum(axis=0)

        if self.X is not None:
            dX = dA.dot(self.W_ih.T)
            self.dW_ih += self.X.T.dot(dA)
        else:
            dX = None

        if self.h_init is not None:
            self.dh_init = self.W_hh.dot(dA[0].reshape(-1,1)).ravel()

        return dX

    def get_h_init_grad(self):
        return self.dh_init if self.h_init is not None else None

    def clear_gradient(self):
        if self.has_input:
            self.dW_ih[:] = 0
        self.dW_hh[:] = 0
        self.db[:] = 0

    def get_param_vec(self):
        if self.has_input:
            return np.r_[self.W_ih.asarray().ravel(), self.W_hh.asarray().ravel(), self.b.asarray().ravel()]
        else:
            return np.r_[self.W_hh.asarray().ravel(), self.b.asarray().ravel()]

    def get_noiseless_param_vec(self):
        return self.get_param_vec()

    def _set_param_from_vec(self, v, is_noiseless=False):
        if self.has_input:
            self.W_ih = gnp.garray(v[:self.W_ih.size].reshape(self.W_ih.shape))
        self.W_hh = gnp.garray(v[-self.W_hh.size-self.b.size:-self.b.size].reshape(self.W_hh.shape))
        self.b = gnp.garray(v[-self.b.size:])

    def get_grad_vec(self):
        if self.has_input:
            return np.r_[self.dW_ih.asarray().ravel(), self.dW_hh.asarray().ravel(), self.db.asarray().ravel()]
        else:
            return np.r_[self.dW_hh.asarray().ravel(), self.db.asarray().ravel()]

    def save_model_to_binary(self):
        s = struct.pack('i', self.get_type_code())
        s += struct.pack('iiii', (1 if self.has_input else 0), (self.in_dim if self.has_input else 0),
                self.out_dim, self.nonlin.get_id())
        if self.has_input:
            s += self.W_ih.asarray().astype(np.float32).tostring()
        s += self.W_hh.asarray().astype(np.float32).tostring()
        s += self.b.asarray().astype(np.float32).tostring()
        return s

    def load_model_from_stream(self, f):
        self.check_type_code(struct.unpack('i', f.read(4))[0])

        has_input, self.in_dim, self.out_dim, nonlin_id = struct.unpack('iiii', f.read(4*4))
        self.has_input = has_input == 1
        if not self.has_input:
            self.in_dim = None

        self.nonlin = layer.get_nonlin_from_type_id(nonlin_id)

        if self.has_input:
            self.W_ih = gnp.garray(np.fromstring(f.read(self.in_dim*self.out_dim*4), 
                dtype=np.float32).reshape(self.in_dim, self.out_dim))
            self.dW_ih = self.W_ih * 0

        self.W_hh = gnp.garray(np.fromstring(f.read(self.out_dim*self.out_dim*4), 
            dtype=np.float32).reshape(self.out_dim, self.out_dim))
        self.b = gnp.garray(np.fromstring(f.read(self.out_dim*4), dtype=np.float32))

        self.dW_hh = self.W_hh * 0
        self.b = self.b * 0

        self._update_param_size()

    def get_type_code(self):
        return 0x0399

    def __repr__(self):
        if self.has_input:
            return 'rnn %d -> %d' % (self.in_dim, self.out_dim)
        else:
            return 'rnn <no input> %d' % self.out_dim

    def _update_param_size(self):
        self.param_size = self.W_hh.size + self.b.size
        if self.has_input:
            self.param_size += self.W_ih.size 

class RnnHybridNetwork(nn.BaseNeuralNet):
    """
    RNN network plus a feed-forward neural net on top of the RNN outputs.
    The RNN network itself can also contain multiple layers, and interleaved
    with feed-forward neural nets.

    TODO:
    - mixing RNN and feed-forward net is to be implemented
    - right now the implementation assumes there is a single RNN at the bottom
      and a neural net on top.
    """
    def __init__(self, rnn=None, feedforward_net=None):
        if rnn is None or feedforward_net is None:
            return

        self.rnn = rnn
        self.feedforward_net = feedforward_net
        self._update_param_size()

        self.in_dim = rnn.in_dim
        self.out_dim = feedforward_net.out_dim

    def load_target(self, *args, **kwargs):
        self.feedforward_net.load_target(*args, **kwargs)

    def get_loss(self):
        return self.feedforward_net.get_loss()

    def forward_prop(self, X=None, T=10, h_init=None, **kwargs):
        """
        options:
        - X can be None, when there's no input, then T must be specified
        - if X is not None, T will not be used
        - an extra h_init can be given to the forward prop to feed into the
          first hidden state activation.
        """
        H = self.rnn.forward_prop(X=X, T=T, h_init=h_init)
        return self.feedforward_net.forward_prop(H, **kwargs)

    def backward_prop(self, grad=None):
        dH = self.feedforward_net.backward_prop(grad=grad)
        return self.rnn.backward_prop(grad=dH)

    def get_h_init_grad(self):
        return self.rnn.get_h_init_grad()

    def clear_gradient(self):
        self.rnn.clear_gradient()
        self.feedforward_net.clear_gradient()

    def get_param_vec(self):
        return np.r_[self.rnn.get_param_vec(), self.feedforward_net.get_param_vec()]

    def get_noiseless_param_vec(self):
        return np.r_[self.rnn.get_noiseless_param_vec(), self.feedforward_net.get_noiseless_param_vec()]

    def _set_param_from_vec(self, v, is_noiseless=False):
        self.rnn._set_param_from_vec(v[:self.rnn.param_size], is_noiseless=is_noiseless)
        self.feedforward_net._set_param_from_vec(v[self.rnn.param_size:], is_noiseless=is_noiseless)

    def get_grad_vec(self):
        return np.r_[self.rnn.get_grad_vec(), self.feedforward_net.get_grad_vec()]

    def save_model_to_binary(self):
        s = struct.pack('i', self.get_type_code())
        s += self.rnn.save_model_to_binary()
        s += self.feedforward_net.save_model_to_binary()
        return s

    def load_model_from_stream(self, f):
        self.check_type_code(struct.unpack('i', f.read(4))[0])

        self.rnn = RNN()
        self.rnn.load_model_from_stream(f)

        self.feedforward_net = nn.NeuralNet()
        self.feedforward_net.load_model_from_stream(f)

        self._update_param_size()

        self.in_dim = self.rnn.in_dim
        self.out_dim = self.feedforward_net.out_dim

    def get_type_code(self):
        return 0x0369

    def __repr__(self):
        return str(self.rnn) + ' | ' + str(self.feedforward_net)

    def _update_param_size(self):
        self.param_size = self.rnn.param_size + self.feedforward_net.param_size

class RnnAutoEncoder(nn.BaseNeuralNet):
    """
    Combination of an encoder RNN and a decoder RnnHybridNetwork.
    """
    def __init__(self, encoder=None, decoder=None):
        if encoder is None or decoder is None:
            return

        self.encoder = encoder
        self.decoder = decoder

        self._update_param_size()

        self.in_dim = encoder.in_dim
        self.out_dim = decoder.out_dim

    def load_target(self, *args, **kwargs):
        pass

    def get_loss(self):
        return self.decoder.get_loss()

    def encode(self, X, h_init=None):
        H = self.encoder.forward_prop(X=X, h_init=h_init)
        return H[-1]

    def forward_prop(self, X, h_init=None, **kwargs):
        """
        options:
        - an extra h_init can be given to the forward prop to feed into the
          first hidden state activation.
        """
        # input is the target
        if kwargs.get('compute_loss', False) == True:
            self.decoder.load_target(X)

        H_encoder = self.encoder.forward_prop(X=X, h_init=h_init)
        return self.decoder.forward_prop(T=X.shape[0], h_init=H_encoder[-1], **kwargs)

    def backward_prop(self, grad=None):
        self.decoder.backward_prop(grad=grad)
        return self.encoder.backward_prop(grad_end=self.decoder.get_h_init_grad())

    def clear_gradient(self):
        self.encoder.clear_gradient()
        self.decoder.clear_gradient()

    def get_param_vec(self):
        return np.r_[self.encoder.get_param_vec(), self.decoder.get_param_vec()]

    def get_noiseless_param_vec(self):
        return np.r_[self.encoder.get_noiseless_param_vec(),
                self.decoder.get_noiseless_param_vec()]

    def _set_param_from_vec(self, v, is_noiseless=False):
        self.encoder._set_param_from_vec(v[:self.encoder.param_size], is_noiseless=is_noiseless)
        self.decoder._set_param_from_vec(v[self.encoder.param_size:], is_noiseless=is_noiseless)

    def get_grad_vec(self):
        return np.r_[self.encoder.get_grad_vec(), self.decoder.get_grad_vec()]

    def save_model_to_binary(self):
        s = struct.pack('i', self.get_type_code())
        s += self.encoder.save_model_to_binary()
        s += self.decoder.save_model_to_binary()
        return s

    def load_model_from_stream(self, f):
        self.check_type_code(struct.unpack('i', f.read(4))[0])

        self.encoder = RNN()
        self.encoder.load_model_from_stream(f)

        self.decoder = RnnHybridNetwork()
        self.decoder.load_model_from_stream(f)

        self._update_param_size()

        self.in_dim = self.encoder.in_dim
        self.out_dim = self.decoder.out_dim

    def get_type_code(self):
        return 0x0363

    def __repr__(self):
        return 'Encoder { ' + str(self.encoder) + ' } Decoder { ' + str(self.decoder) + ' }'

    def _update_param_size(self):
        self.param_size = self.encoder.param_size + self.decoder.param_size



