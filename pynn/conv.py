"""
Convolutional layer.

This implementation is incomplete.

Yujia Li, 05/2015
"""

import struct
import numpy as np
import gnumpy as gnp
import preprocessor as pp
import clustering as clust

class ConvShape(object):
    """
    This explains how the data is stored.  Each image is assumed to be stored
    first by channel, in each channel pixels organized in row-major order.
    """
    def __init__(self, height, width, channels):
        self.h = height
        self.w = width
        self.c = channels

    def size(self):
        return self.h * self.w * self.c

class FixedConvolutionalLayer(object):
    """
    Fixed convolutional layers do not have trainable parameters.  To use them,
    train a model offline, then load it and use it to make predictions.
    """
    def __init__(self, n_ic=None, n_oc=None, ksize=5, stride=2, prep=None):
        """
        n_ic: number of input channels
        n_oc: number of output channels
        ksize: size of the kernel
        stride: stride
        prep: preprocessor
        """
        if n_ic is None or n_oc is None:
            return

        self.n_ic = n_ic
        self.n_oc = n_oc
        self.ksize = ksize
        self.stride = stride
        self.prep = prep

    def compute_output_shape(self, input_data_shape):
        return ConvShape(
                (input_data_shape.h - self.ksize + self.stride) / self.stride,
                (input_data_shape.w - self.ksize + self.stride) / self.stride,
                self.n_oc)

    def extract_patches(self, X, data_shape):
        """
        Extract patches from input data according to its shape and the kernel
        configurations.

        The patches are ordered by height-width-nimages
        """
        # a copy is necessary to make the strides easy to index
        X = gnp.as_numpy_array(X).reshape(-1, data_shape.c, data_shape.h, data_shape.w)

        assert data_shape.c == self.n_ic

        patches = []
        for i in xrange(0, data_shape.h - self.ksize + 1, self.stride):
            for j in xrange(0, data_shape.w - self.ksize + 1, self.stride):
                patches.append(X[:,:,i:i+self.ksize, j:j+self.ksize])

        return np.concatenate(patches, axis=0).reshape(-1, self.ksize*self.ksize*self.n_ic)

    def reorganize_patch_responses(self, r, data_shape):
        """
        r: (H*W*N)xC matrix.
        """
        return r.reshape(data_shape.h, data_shape.w, -1, data_shape.c).transpose((2,3,0,1)).reshape(-1, data_shape.size())

    def compute_patch_responses(self, patches):
        raise NotImplementedError()

    def forward_prop(self, X, data_shape):
        """
        Return output and output_shape
        """
        output_shape = self.compute_output_shape(data_shape)
        P = self.extract_patches(X, data_shape)

        assert P.shape[0] == X.shape[0] * output_shape.w * output_shape.h
        assert P.shape[1] == self.ksize * self.ksize * self.n_ic

        R = self.compute_patch_responses(P)

        assert R.shape[1] == output_shape.c

        return self.reorganize_patch_responses(R, output_shape), output_shape

    def recover_input(self, Y, out_shape, in_shape):
        """
        Return recovered input and input_shape
        """
        Y = gnp.as_numpy_array(Y).reshape(-1, out_shape.c, out_shape.h, out_shape.w).transpose((0,2,3,1)).reshape(-1, out_shape.c)
        P = self.recover_patches_from_responses(Y)
        return self.overlay_patches(P, out_shape, in_shape)

    def recover_patches_from_responses(self, resp):
        raise NotImplementedError()

    def overlay_patches(self, patches, out_shape, in_shape):
        """
        patches are assumed to be organized as a (NxHxW)*C matrix.
        """
        P = patches.reshape(-1, out_shape.h, out_shape.w, in_shape.c, self.ksize, self.ksize).transpose((0,3,1,2,4,5))
        # P = patches.reshape(-1, out_shape.h, out_shape.w, in_shape.c).transpose((0,3,1,2))

        X = np.zeros((P.shape[0], in_shape.c, in_shape.h, in_shape.w), dtype=np.float32)
        overlay_count = X * 0

        for i in xrange(out_shape.h):
            for j in xrange(out_shape.w):
                X[:,:,i*self.stride:i*self.stride+self.ksize,j*self.stride:j*self.stride+self.ksize] += P[:,:,i,j,:,:]
                overlay_count[:,:,i*self.stride:i*self.stride+self.ksize,j*self.stride:j*self.stride+self.ksize] += 1

        return X / (overlay_count + 1e-10)
    
    @staticmethod
    def load_model_from_stream(f):
        type_code = struct.unpack('i', f.read(4))[0]
        if type_code == KMeansLayer.get_type_code():
            layer = KMeansLayer()
        else:
            raise Exception('Type code "%d" not recognized.' % type_code)
        layer._load_model_from_stream(f)
        return layer

    def _load_model_from_stream(self, f):
        self.n_ic, self.n_oc, self.ksize, self.stride = struct.unpack('iiii', f.read(4*4))
        self.prep = pp.Preprocessor.load_from_stream(f)

    def save_model_to_binary(self):
        return struct.pack('i', self.get_type_code()) + self._save_model_to_binary()

    def _save_model_to_binary(self):
        return struct.pack('iiii', n_ic, n_oc, ksize, stride) + prep.save_to_binary()

    @staticmethod
    def get_type_code():
        raise NotImplementedError()

class KMeansLayer(FixedConvolutionalLayer):
    def __init__(self, C=None, dist='euclidean', n_ic=None, n_oc=None, ksize=5, stride=2):
        """
        C: KxD matrix
        """
        if C is None:
            return
        self.C = gnp.as_garray(C)
        n_oc = C.shape[0]
        super(KMeansLayer, self).__init__(n_ic=n_ic, n_oc=n_oc, ksize=ksize, stride=stride)
        assert C.shape[0] == ksize*ksize*n_ic

        self.f_dist = clust.choose_distance_metric(dist)

    def _save_model_to_binary(self):
        return super(KMeansLayer, self)._save_model_to_binary() + \
                self.C.asarray().astype(np.float32).ravel().tostring()

    def _load_model_from_stream(self, f):
        super(KMeansLayer, self)._load_model_from_stream()

        K = self.n_oc
        D = self.n_ic * self.ksize * self.ksize

        self.C = gnp.garray(np.fromstring(f.read(4*K*D), dtype=np.float32).reshape(
            self.n_oc, self.n_ic*self.ksize*self.ksize))

    @staticmethod
    def get_type_code():
        return 0

    def compute_patch_responses(self, patches):
        D = self.f_dist(patches, self.C)
        idx = D.argmin(axis=1)
        R = np.zeros((patches.shape[0], self.n_oc), dtype=np.float32)
        R[np.arange(R.shape[0]), idx] = 1
        return R

class TriangleKMeansLayer(KMeansLayer):
    """
    A softer version of k-means introduced in Coates, Lee and Ng (2011).
    """
    @staticmethod
    def get_type_code():
        return 1

    def compute_patch_responses(self, patches):
        D = self.f_dist(patches, self.C)
        d_avg = D.mean(axis=0)
        R = (d_avg.reshape(1,-1) - D)
        return R * (R > 0)

