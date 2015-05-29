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
    def __init__(self, height=0, width=0, channels=0):
        self.h = height
        self.w = width
        self.c = channels

    def size(self):
        return self.h * self.w * self.c

    def save_to_binary(self):
        return struct.pack('iii', self.h, self.w, self.c)

    def load_from_stream(self, f):
        self.h, self.w, self.c = struct.unpack('iii', f.read(3*4))

    def __repr__(self):
        return '<ConvShape h=%d, w=%d, c=%d, size=%d>' % (self.h, self.w, self.c, self.size())

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
        """
        The computed output shape also takes care of the padding to avoid boundary effects
        """
        return ConvShape(
                (input_data_shape.h - self.ksize + self.stride - 1) / self.stride + 1,
                (input_data_shape.w - self.ksize + self.stride - 1) / self.stride + 1,
                self.n_oc)

    def extract_patches(self, X, data_shape):
        """
        Extract patches from input data according to its shape and the kernel
        configurations.

        Return patches matrix of size (H*W*N)x(C*ksize*ksize)
        """
        X = gnp.as_numpy_array(X).reshape(-1, data_shape.c, data_shape.h, data_shape.w)

        out_shape = self.compute_output_shape(data_shape)
        padded_h = (out_shape.h - 1) * self.stride + self.ksize
        padded_w = (out_shape.w - 1) * self.stride + self.ksize

        if padded_h > data_shape.h or padded_w > data_shape.w:
            new_X = np.zeros((X.shape[0], X.shape[1], padded_h, padded_w), dtype=X.dtype)
            new_X[:,:,:data_shape.h, :data_shape.w] = X
            X = new_X
        
        assert data_shape.c == self.n_ic

        patches = []
        for i in xrange(0, X.shape[-2] - self.ksize + 1, self.stride):
            for j in xrange(0, X.shape[-1] - self.ksize + 1, self.stride):
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

    def recover_input(self, Y, out_shape, in_shape, **kwargs):
        """
        Return recovered input and input_shape
        """
        Y = gnp.as_numpy_array(Y).reshape(-1, out_shape.c, out_shape.h, out_shape.w).transpose((0,2,3,1)).reshape(-1, out_shape.c)
        P = self.recover_patches_from_responses(Y, **kwargs)
        return self.overlay_patches(P, out_shape, in_shape)

    def recover_patches_from_responses(self, resp, **kwargs):
        """
        resp: (N*H*W)xC_out matrix

        Return (N*H*W)x(C_in*ksize*ksize) matrix
        """
        raise NotImplementedError()

    def compute_input_shape(self, out_shape):
        return ConvShape(
                (out_shape.h - 1) * self.stride + self.ksize,
                (out_shape.w - 1) * self.stride + self.ksize,
                self.n_ic)

    def overlay_patches(self, patches, out_shape, in_shape):
        """
        patches are assumed to be organized as a (NxHxW)*C matrix.
        """
        P = patches.reshape(-1, out_shape.h, out_shape.w, in_shape.c, self.ksize, self.ksize).transpose((0,3,1,2,4,5))

        padded_in_shape = self.compute_input_shape(out_shape)
        assert padded_in_shape.c == in_shape.c

        X = np.zeros((P.shape[0], padded_in_shape.c, padded_in_shape.h, padded_in_shape.w), dtype=np.float32)
        overlay_count = X * 0

        for i in xrange(out_shape.h):
            for j in xrange(out_shape.w):
                X[:,:,i*self.stride:i*self.stride+self.ksize,j*self.stride:j*self.stride+self.ksize] += P[:,:,i,j,:,:]
                overlay_count[:,:,i*self.stride:i*self.stride+self.ksize,j*self.stride:j*self.stride+self.ksize] += 1

        X /= (overlay_count + 1e-10)
        X = X[:,:,:in_shape.h,:in_shape.w]
        return X.reshape(X.shape[0], -1)
    
    @staticmethod
    def load_model_from_stream(f):
        type_code = struct.unpack('i', f.read(4))[0]
        if type_code == KMeansLayer.get_type_code():
            layer = KMeansLayer()
        elif type_code == TriangleKMeansLayer.get_type_code():
            layer = TriangleKMeansLayer()
        else:
            raise Exception('Type code "%d" not recognized.' % type_code)
        layer._load_model_from_stream(f)
        return layer

    def _load_model_from_stream(self, f):
        self.n_ic, self.n_oc, self.ksize, self.stride = struct.unpack('iiii', f.read(4*4))
        if struct.unpack('i', f.read(4))[0] == 1:
            self.prep = pp.Preprocessor.load_from_stream(f)
        else:
            self.prep = None

    def save_model_to_binary(self):
        return struct.pack('i', self.get_type_code()) + self._save_model_to_binary()

    def _save_model_to_binary(self):
        s = struct.pack('iiii', self.n_ic, self.n_oc, self.ksize, self.stride)
        if self.prep:
            s += struct.pack('i', 1) + self.prep.save_to_binary()
        else:
            s += struct.pack('i', 0)
        return s

    @staticmethod
    def get_type_code():
        raise NotImplementedError()

    @staticmethod
    def get_type_name():
        raise NotImplementedError()

    def __repr__(self):
        return '%s %d-%dx%dx%d' % (self.get_type_name(), self.n_oc, self.ksize, self.ksize, self.n_ic)


class KMeansLayer(FixedConvolutionalLayer):
    def __init__(self, kmeans_model=None, stride=2):
        """
        kmeans_model: instance of KMeansModel, contains attributes (C, dist, nc, ksize, prep)
        """
        if kmeans_model is None:
            return

        self.C = gnp.as_garray(kmeans_model.C)
        dist = kmeans_model.dist
        n_ic = kmeans_model.nc
        n_oc = kmeans_model.C.shape[0]
        ksize = kmeans_model.ksize
        prep = kmeans_model.prep

        super(KMeansLayer, self).__init__(n_ic=n_ic, n_oc=n_oc, ksize=ksize, stride=stride, prep=prep)
        assert self.C.shape[1] == ksize*ksize*n_ic

        self.f_dist = clust.choose_distance_metric(dist)
        self.dist = dist

    def _save_model_to_binary(self):
        return super(KMeansLayer, self)._save_model_to_binary() + \
                self.C.asarray().astype(np.float32).ravel().tostring() + \
                struct.pack('i', len(self.dist)) + str(self.dist)

    def _load_model_from_stream(self, f):
        super(KMeansLayer, self)._load_model_from_stream(f)

        K = self.n_oc
        D = self.n_ic * self.ksize * self.ksize

        self.C = gnp.garray(np.fromstring(f.read(4*K*D), dtype=np.float32).reshape(
            self.n_oc, self.n_ic*self.ksize*self.ksize))

        dist_name_len = struct.unpack('i', f.read(4))[0]
        self.dist = f.read(dist_name_len)
        self.f_dist = clust.choose_distance_metric(self.dist)

    @staticmethod
    def get_type_code():
        return 0

    @staticmethod
    def get_type_name():
        return 'kmeans'

    def compute_patch_responses(self, patches):
        if self.prep is not None:
            patches = self.prep.process(patches)
        patches = gnp.as_garray(patches)
        D = self.f_dist(patches, self.C).asarray().astype(np.float64)
        idx = D.argmin(axis=1)
        R = np.zeros((patches.shape[0], self.n_oc), dtype=np.float32)
        R[np.arange(R.shape[0]), idx] = 1
        return R

    def recover_patches_from_responses(self, resp, hard_assignment=False):
        if hard_assignment:
            P = self.C[resp.argmax(axis=1)]
        else:
            P = gnp.as_garray(resp).dot(self.C)
        if self.prep is not None:
            P = self.prep.reverse(P)
        return P.asarray()

class TriangleKMeansLayer(KMeansLayer):
    """
    A softer version of k-means introduced in Coates, Lee and Ng (2011).
    """
    @staticmethod
    def get_type_code():
        return 1

    @staticmethod
    def get_type_name():
        return 'triangle-kmeans'

    def compute_patch_responses(self, patches):
        if self.prep is not None:
            patches = self.prep.process(patches)
        patches = gnp.as_garray(patches)
        D = self.f_dist(patches, self.C).asarray().astype(np.float64)
        d_avg = D.mean(axis=0)
        R = (d_avg.reshape(1,-1) - D)
        return R * (R > 0)

    def recover_patches_from_responses(self, resp):
        R = gnp.as_garray(resp)
        R /= R.sum(axis=1).reshape(-1,1)
        P = R.dot(self.C)
        if self.prep is not None:
            P = self.prep.reverse(P)
        return P.asarray()

class KMeansModel(object):
    def __init__(self, C, dist, nc, ksize, prep):
        self.C = C
        self.dist = dist
        self.nc = nc
        self.ksize = ksize
        self.prep = prep

def get_random_patches(X, in_shape, ksize, n_patches_per_image, batch_size=100, pad_h=0, pad_w=0):
    """
    Extract random patches from images X.

    X: Nx(C*H*W) matrix, each row is an image
    in_shape: shape information for each input image
    ksize: size of the patches
    n_patches_per_image: number of patches per image
    batch_size: size of a batch.  In each batch the patch locations will be the
        same.

    Return (n_patches_per_image*N)x(C*ksize*ksize) matrix, each row is one
        patch.
    """
    X = gnp.as_numpy_array(X).reshape(-1, in_shape.c, in_shape.h, in_shape.w)
    if pad_h > 0 or pad_w > 0:
        new_X = np.zeros((X.shape[0], in_shape.c, in_shape.h + pad_h, in_shape.w + pad_w), dtype=X.dtype)
        new_X[:,:,:in_shape.h,:in_shape.w] = X
        X = new_X

    patches = []
    for n in xrange(n_patches_per_image):
        for im_idx in xrange(0, X.shape[0], batch_size):
            h_start = np.random.randint(X.shape[-2] - ksize + 1)
            w_start = np.random.randint(X.shape[-1] - ksize + 1)

            patches.append(X[im_idx:im_idx+batch_size,:,h_start:h_start+ksize,w_start:w_start+ksize])

    return np.concatenate(patches, axis=0).reshape(-1, in_shape.c*ksize*ksize)

def train_kmeans_layer(X, in_shape, K, ksize, n_patches_per_image, prep_type=None, pad_h=0, pad_w=0, repeat=1, **kwargs):
    train_data = get_random_patches(X, in_shape, ksize, n_patches_per_image, pad_h=pad_h, pad_w=pad_w)
    if prep_type is not None:
        prep = pp.choose_preprocessor_by_name(prep_type)
        prep.train(train_data)
        train_data = prep.process(prep)
    else:
        prep = None

    C_best = None
    loss_best = None

    for i_repeat in xrange(repeat):
        print '*** repeat #%d ***' % (i_repeat + 1)
        gnp.free_reuse_cache()
        C, _, loss = clust.kmeans(train_data, K, **kwargs) 
        if loss_best is None or loss < loss_best:
            loss_best = loss
            C_best = C

    print '>>> best loss: %.2f' % loss_best
    return KMeansModel(C_best, kwargs.get('dist', 'euclidean'), in_shape.c, ksize, prep)

class FixedConvolutionalNetwork(object):
    """
    Multiple fixed convolutional layers stacked on top of each other.
    """
    def __init__(self, layers=[]):
        self.layers = layers[:]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_prop(self, X, data_shape):
        for l in self.layers:
            X, data_shape = l.forward_prop(X, data_shape)
        return X, data_shape
    
    def recover_input(self, Y, out_shape, in_shape, **kwargs):
        """
        If in_shapes is None, this will try to use the in_shapes from a
        previous forward prop.
        """
        in_shapes = []
        for l in self.layers:
            in_shapes.append(in_shape)
            in_shape = l.compute_output_shape(in_shape)

        for i in range(len(self.layers))[::-1]:
            Y = self.layers[i].recover_input(Y, out_shape, in_shapes[i], **kwargs)
            out_shape = in_shapes[i]

        return Y

    def load_model_from_stream(self, f):
        n_layers = struct.unpack('i', f.read(4))[0]
        self.layers = []
        for i in xrange(n_layers):
            self.layers.append(FixedConvolutionalLayer.load_model_from_stream(f))

    def save_model_to_binary(self):
        s = struct.pack('i', len(self.layers)) + ''.join([l.save_model_to_binary() for l in self.layers])
        return s

    def load_model_from_file(self, fname):
        with open(fname, 'rb') as f:
            self.load_model_from_stream(f)

    def save_model_to_file(self, fname):
        with open(fname, 'wb') as f:
            f.write(self.save_model_to_binary())

    def __repr__(self):
        return ' | '.join([str(l) for l in self.layers])

def build_kmeans_convnet(X, in_shape, layer_configs=[], n_patches_per_image=100, kmeans_repeat=1, prep_type=None,
        use_triangle_kmeans=False, **kwargs):
    """
    X: Nx(C*H*W) image matrix, each row is an image
    in_shape: shape of the input image
    layer_configs: a list of layer configurations, each element is a (K, 
        ksize, stride, n_patches_per_image, prep_type) or (K, ksize, stride) tuple.  If 
        n_patches_per_image and prep_type are not set in the tuples, global
        values will be used to train all layers.
    n_patches_per_image: only used when layer_configs is a list of 2-tuples
    prep_type: only used when layer_configs is a list of 2-tuples
    use_triangle_kmeans: use TriangleKMeansLayer instead if True
    kwargs: extra arguments for k-means training.
    """
    if len(layer_configs) == 0:
        return

    kmnn = FixedConvolutionalNetwork()
    for i in xrange(len(layer_configs)):
        if len(layer_configs[i]) == 5:
            K, ksize, stride, n_patches_per_image_, prep_type_ = layer_configs[i]
        else:
            K, ksize, stride = layer_configs[i][:3]
            n_patches_per_image_ = n_patches_per_image
            prep_type_ = prep_type

        if i == 0:
            train_data = X
            train_in_shape = in_shape
        else:
            train_data, train_in_shape = kmnn.layers[-1].forward_prop(train_data, train_in_shape)

        print ''
        print '===== Training layer %d =====' % (i+1)
        print 'data shape: %s' % str(train_in_shape)
        print 'current network: %s' % str(kmnn)
        print ''

        pad_h = (stride - (train_in_shape.h - ksize) % stride) % stride
        pad_w = (stride - (train_in_shape.w - ksize) % stride) % stride

        m = train_kmeans_layer(train_data, train_in_shape, K, ksize, 
                n_patches_per_image_, prep_type_, pad_h=pad_h, pad_w=pad_w, repeat=kmeans_repeat, **kwargs)
        kmnn.add_layer(KMeansLayer(m, stride=stride) if not use_triangle_kmeans \
                else TriangleKMeansLayer(m, stride=stride))

    print 'Final network: ' + str(kmnn)
    return kmnn


