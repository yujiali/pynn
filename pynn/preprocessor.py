"""
Data preprocessors.

Yujia Li, 05/2015
"""

import gnumpy as gnp
import numpy as np
import scipy.linalg as la
import struct
_SMALL_CONSTANT = 1e-8

def choose_preprocessor_by_name(prep_name, *args, **kwargs):
    if prep_name == 'blank':
        return BlankPreprocessor(*args, **kwargs)
    elif prep_name == 'mean_std':
        return MeanStdPreprocessor(*args, **kwargs)
    elif prep_name == 'std_norm':
        return StdNormPreprocessor(*args, **kwargs)
    elif prep_name == 'whitening':
        return WhiteningPreprocessor(*args, **kwargs)
    elif prep_name == 'pca':
        return PCAPreprocessor(*args, **kwargs)
    else:
        raise Exception('Preprocessor name "%s" not recognized.' % prep_name)

class Preprocessor(object):
    """Base class for preprocessors."""
    def __init__(self, x=None, prev=None, **kwargs):
        """Construct preprocessor.  Can use some data x, or chained with
        another preprocessor."""
        self.prev = prev
        if x is None:
            return
        else:
            self.train(x)

    def train(self, x):
        """
        This learns the parameters of the preprocessor from data.
        """
        pass

    def process(self, x):
        """Process data x and return a processed copy x_new."""
        pass

    def reverse(self, processed_x):
        """Recover the original x from processed x."""
        pass

    @staticmethod
    def get_type_code():
        """
        Each preprocessor has a unique type code.
        """
        raise NotImplementedError()

    def save_to_binary(self):
        return struct.pack('i', self.get_type_code()) + self._save_to_binary()

    def _save_to_binary(self):
        raise NotImplementedError()

    def save_to_file(self, fname):
        with open(fname, 'wb') as f:
            f.write(self.save_to_binary())

    @staticmethod
    def load_from_stream(f):
        type_code = struct.unpack('i', f.read(4))[0]
        if type_code == BlankPreprocessor.get_type_code():
            prep = BlankPreprocessor()
        elif type_code == MeanStdPreprocessor.get_type_code():
            prep = MeanStdPreprocessor()
        elif type_code == StdNormPreprocessor.get_type_code():
            prep = StdNormPreprocessor()
        elif type_code == WhiteningPreprocessor.get_type_code():
            prep = WhiteningPreprocessor()
        elif type_code == PCAPreprocessor.get_type_code():
            prep = PCAPreprocessor()
        else:
            raise Exception('Type code "%d" for Preprocessor not recognized.' % type_code)

        prep._load_from_stream(f)
        return prep

    def _load_from_stream(self, f):
        raise NotImplementedError()

    @staticmethod
    def load_from_file(fname):
        with open(fname, 'rb') as f:
            return Preprocessor.load_from_stream(f)

class BlankPreprocessor(Preprocessor):
    """Do nothing."""
    def train(self, x):
        pass

    def process(self, x):
        return x

    def reverse(self, processed_x):
        return x

    @staticmethod
    def get_type_code():
        return 0

    def _save_to_binary(self):
        return ''

    def _load_from_stream(self, f):
        pass

class MeanStdPreprocessor(Preprocessor):
    """Subtract mean and normalize by standard deviation preprocessor."""
    def train(self, x):
        if self.prev:
            x = self.prev.process(x)
        self.avg = x.mean(axis=0)
        self.std = x.std(axis=0) + _SMALL_CONSTANT

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return (x - self.avg) / self.std

    def reverse(self, processed_x):
        x = processed_x * self.std + self.avg
        if self.prev:
            return self.prev.reverse(x)
        else:
            return x

    @staticmethod
    def get_type_code():
        return 1

    def _save_to_binary(self):
        s = ''
        if self.prev is not None:
            s += struct.pack('i', 1)
            s += self.prev.save_to_binary()
        else:
            s += struct.pack('i', 0)

        s += struct.pack('i', self.avg.size)
        s += self.avg.astype(np.float32).tostring()
        s += self.std.astype(np.float32).tostring()
        return s

    def _load_from_stream(self, f):
        if struct.unpack('i', f.read(4))[0] == 1:
            self.prev = Preprocessor.load_from_stream(f)
        else:
            self.prev = None

        D = struct.unpack('i', f.read(4))[0]
        self.avg = np.fromstring(f.read(4*D), dtype=np.float32)
        self.std = np.fromstring(f.read(4*D), dtype=np.float32)

class StdNormPreprocessor(Preprocessor):
    """Normalize the features using standard deviation."""
    def train(self, x):
        if self.prev:
            x = self.prev.process(x)
        self.std = x.std(axis=0) + _SMALL_CONSTANT

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return x / self.std

    def reverse(self, processed_x):
        x = processed_x * self.std
        if self.prev:
            return self.prev.reverse(x)
        else:
            return x

    @staticmethod
    def get_type_code():
        return 2

    def _save_to_binary(self):
        s = ''
        if self.prev is not None:
            s += struct.pack('i', 1)
            s += self.prev.save_to_binary()
        else:
            s += struct.pack('i', 0)

        s += struct.pack('i', self.std.size)
        s += self.std.astype(np.float32).tostring()
        return s

    def _load_from_stream(self, f):
        if struct.unpack('i', f.read(4))[0] == 1:
            self.prev = Preprocessor.load_from_stream(f)
        else:
            self.prev = None

        D = struct.unpack('i', f.read(4))[0]
        self.std = np.fromstring(f.read(4*D), dtype=np.float32)

class WhiteningPreprocessor(Preprocessor):
    """Whitening - decorrelate covariance."""
    def train(self, x):
        if self.prev:
            x = self.prev.process(x)
        x = gnp.as_garray(x)
        self.avg = x.mean(axis=0)
        cov = (x - self.avg).T.dot(x - self.avg) / x.shape[0]
        cov = gnp.as_numpy_array(cov)
        self.sqrcov = la.cholesky(cov + np.eye(cov.shape[0]) * 1e-5)
        self.m = gnp.as_garray(la.inv(self.sqrcov + np.eye(x.shape[1]) * 1e-5))

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        x = gnp.as_garray(x)
        return (x - self.avg).dot(self.m)

    def reverse(self, processed_x):
        processed_x = gnp.as_garray(processed_x)
        sqrcov = gnp.as_garray(self.sqrcov)
        x = processed_x.dot(sqrcov) + self.avg
        if self.prev:
            return self.prev.reverse(x)
        else:
            return x

    @staticmethod
    def get_type_code():
        return 3

    def _save_to_binary(self):
        s = ''
        if self.prev is not None:
            s += struct.pack('i', 1)
            s += self.prev.save_to_binary()
        else:
            s += struct.pack('i', 0)

        s += struct.pack('i', self.avg.size)
        s += self.avg.asarray().astype(np.float32).tostring()
        s += self.m.asarray().astype(np.float32).ravel().tostring()
        return s

    def _load_from_stream(self, f):
        if struct.unpack('i', f.read(4))[0] == 1:
            self.prev = Preprocessor.load_from_stream(f)
        else:
            self.prev = None

        D = struct.unpack('i', f.read(4))[0]
        self.avg = gnp.garray(np.fromstring(f.read(4*D), dtype=np.float32))
        self.m = gnp.garray(np.fromstring(f.read(4*D*D), dtype=np.float32).reshape(D,D))

def pca(x, K):
    """(x, K) --> (xnew, basis, xmean)

    x: N*D is the data matrix, each row is a data vector
    K: an integer, the dimensionality of the low dimensional space to project

    xnew: N*K projected data matrix
    basis: D*K matrix, each column is a basis vector for the low dimensional space
    xmean: 1-D vector, the mean vector of x
    """

    xmean = x.mean(axis=0)

    X = x - xmean
    [w, basis] = np.linalg.eigh(X.T.dot(X))
    idx = np.argsort(w)
    idx = idx[::-1]
    basis = basis[:,idx[:K]]

    xnew = X.dot(basis)

    return xnew, basis, xmean

def pca_dim_reduction(x, basis, xmean=None):
    """(x, basis, xmean) --> xnew
    Dimensionality reduction with PCA.

    x: N*D data matrix
    basis: D*K basis matrix
    xmean: 1-D vector, mean vector used in PCA, if not set, use the mean of x instead

    xnew: N*K new data matrix
    """

    if xmean is None:
        xmean = x.mean(axis=0)

    xnew = (x - xmean).dot(basis)
    return xnew

def pca_reconstruction(x, basis, xmean=None):
    """Map x in the PCA space back to the original space using the basis and
    mean of the PCA mapping."""
    xrec = x.dot(basis.T)
    if xmean is not None:
        xrec += xmean
    return xrec

class PCAPreprocessor(Preprocessor):
    """PCA"""
    def __init__(self, x=None, prev=None, K=None):
        self.prev = prev
        self.K = K

        if x is None:
            return
        else:
            self.train(x)

    def train(self, x):
        if self.prev:
            x = self.prev.process(x)
        if self.K == None:
            self.K = x.shape[1] / 2 + 1
        _, self.basis, self.avg = pca(x, self.K)

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return pca_dim_reduction(x, self.basis, self.avg)

    def reverse(self, processed_x):
        x = pca_reconstruction(processed_x, self.basis, self.avg)
        if self.prev:
            return self.prev.reverse(x)
        else:
            return x

    @staticmethod
    def get_type_code():
        return 4

    def _save_to_binary(self):
        s = ''
        if self.prev is not None:
            s += struct.pack('i', 1)
            s += self.prev.save_to_binary()
        else:
            s += struct.pack('i', 0)

        s += struct.pack('i', self.K)
        s += struct.pack('i', self.avg.size)
        s += self.avg.astype(np.float32).tostring()
        s += self.basis.astype(np.float32).ravel().tostring()
        return s

    def _load_from_stream(self, f):
        if struct.unpack('i', f.read(4))[0] == 1:
            self.prev = Preprocessor.load_from_stream(f)
        else:
            self.prev = None

        K = struct.unpack('i', f.read(4))[0]
        D = struct.unpack('i', f.read(4))[0]
        self.avg = np.fromstring(f.read(4*D), dtype=np.float32)
        self.basis = np.fromstring(f.read(4*D*K), dtype=np.float32).reshape(D,K)

