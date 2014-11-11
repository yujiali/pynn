"""
Learners updates the model and handles data I/O.

Yujia Li, 09/2014
"""
import pyopt.opt as opt
import color as co
import numpy as np
import gnumpy as gnp
import scipy.optimize as spopt
import os

class ParamCache(object):
    """
    This class implements a parameter cache that keeps track of a small history
    of past parameters.
    """
    def __init__(self, param_size, history_size):
        self.cache = np.empty((param_size, history_size), dtype=np.float)
        self.init_period = True
        self.cache_ptr = 0
        self.cache_size = history_size

    def add_param(self, w):
        self.cache[:, self.cache_ptr] = w
        self.cache_ptr = (self.cache_ptr + 1) % self.cache_size
        if self.init_period and self.cache_ptr == 0:
            self.init_period = False

    def get_average_param(self):
        if self.init_period:
            return self.cache[:, :self.cache_ptr].mean(axis=1)
        else:
            return self.cache.mean(axis=1)

class MiniBatchGenerator(object):
    """
    Generate minibatches, useful for stochastic gradient descent learning.
    """
    def __init__(self, x, t=None, minibatch_size=100, random_order=True): 
        """
        Both x and t are arrays of length n_cases. Target t may be None.
        """
        self.x = x
        self.t = t
        self.minibatch_size = minibatch_size
        self.n_cases = x.shape[0]
        self.random_order = random_order
        self.shuffle_data()

    def __iter__(self):
        return self

    def shuffle_data(self):
        if self.random_order:
            self.idx = np.random.permutation(self.n_cases)
        else:
            self.idx = np.arange(self.n_cases)
        self.i_ptr = 0

    def next(self):
        """
        Get the next minibatch of data.

        Return a tuple of (minibatch_x, minibatch_t) if t is not None,
        otherwise return only minibatch_x.
        """
        minibatch_t = None
        if self.i_ptr + self.minibatch_size <= self.n_cases:
            minibatch_x = self.x[self.idx[self.i_ptr:self.i_ptr + self.minibatch_size]]
            if self.t is not None:
                minibatch_t = self.t[self.idx[self.i_ptr:self.i_ptr + self.minibatch_size]]
            self.i_ptr += self.minibatch_size
        else:
            minibatch_x_part = self.x[self.idx[self.i_ptr:]].copy()
            if self.t is not None:
                minibatch_t_part = self.t[self.idx[self.i_ptr:]].copy()

            other_part_size = self.minibatch_size - (self.n_cases - self.i_ptr)
            self.shuffle_data()
            if isinstance(self.x, gnp.garray):
                minibatch_x = gnp.concatenate([minibatch_x_part, self.x[self.idx[:other_part_size]]], axis=0)
            else:
                minibatch_x = np.r_[minibatch_x_part, self.x[self.idx[:other_part_size]]]

            if self.t is not None:
                if isinstance(self.t, gnp.garray):
                    minibatch_t = gnp.concatenate([minibatch_t_part, self.t[self.idx[:other_part_size]]], axis=0)
                else:
                    minibatch_t = np.r_[minibatch_t_part, self.t[self.idx[:other_part_size]]]

            self.i_ptr = other_part_size

        if self.t is not None:
            return minibatch_x, minibatch_t
        else:
            return minibatch_x

class Learner(object):
    """
    Base class for all learners.
    """
    def __init__(self, net, param_cache_size=1):
        """
        net is a BaseNeuralNet instance.
        """
        self.net = net
        self._init_param_cache(param_cache_size)
        self.output_dir = '.'

    def _init_param_cache(self, param_cache_size):
        self.param_cache_size = param_cache_size
        if param_cache_size > 1:
            self.param_cache = ParamCache(self.net.param_size, param_cache_size)

    def set_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def load_data(self, x_train, t_train, x_val=None, t_val=None):
        """
        Load training and validation data.
        """
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val

        self.use_validation = (self.x_val is not None) and (self.t_val is not None)

    def load_train_target(self):
        """
        Load targets for training.
        """
        self.net.load_target(self.t_train)

    def f_and_fprime(self, w):
        self.net.set_param_from_vec(w)
        self.net.clear_gradient()
        self.net.forward_prop(self.x_train, add_noise=True, compute_loss=True)
        loss = self.net.get_loss()
        self.net.backward_prop()
        grad = self.net.get_grad_vec()
        return loss, grad

    def create_minibatch_generator(self, minibatch_size):
        """
        Can be customized.
        """
        self.minibatch_generator = MiniBatchGenerator(
                self.x_train, t=self.t_train, minibatch_size=minibatch_size,
                random_order=True)

    def f_and_fprime_minibatch(self, w):
        x, t = self.minibatch_generator.next()

        self.net.set_param_from_vec(w)
        self.net.clear_gradient()
        self.net.load_target(t)
        self.net.forward_prop(x, add_noise=True, compute_loss=True)
        loss = self.net.get_loss()
        self.net.backward_prop()
        grad = self.net.get_grad_vec()

        return loss, grad

    def get_f_and_fprime_func(self, weight_decay=0):
        if weight_decay == 0:
            return self.f_and_fprime
        else:
            def f_and_fprime(w):
                loss, grad = self.f_and_fprime(w)
                loss += 0.5 * weight_decay * (w**2).sum()
                grad += weight_decay * w
                return loss, grad
            return f_and_fprime

    def f_info(self, w):
        """
        This is a reference implementatoin of this function, but can be 
        customized for other learners as well.
        """
        train_loss = None
        val_loss = None

        w_0 = self.net.get_param_vec()

        self.net.set_noiseless_param_from_vec(w)
        self.net.load_target(self.t_train)
        self.net.forward_prop(self.x_train, add_noise=False, compute_loss=True)
        train_loss = self.net.get_loss() / self.x_train.shape[0]

        if self.use_validation:
            self.net.load_target(self.t_val)
            self.net.forward_prop(self.x_val, add_noise=False, compute_loss=True)
            val_loss = self.net.get_loss() / self.x_val.shape[0]
            self.net.load_target(self.t_train)

            s = 'train loss %.4f, val loss ' % train_loss
            if self.best_obj is None or val_loss < self.best_obj:
                self.best_obj = val_loss
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % val_loss)
            else:
                s += '%.4f' % val_loss
        else:
            s = 'train loss '
            if self.best_obj is None or train_loss < self.best_obj:
                self.best_obj = train_loss
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % train_loss)
            else:
                s += '%.4f' % train_loss

        self.net.set_param_from_vec(w_0)
        return s

    def _f_info_decorated(self, w):
        if self.param_cache_size == 1:
            return self.f_info(w)
        else:
            self.param_cache.add_param(w)
            return self.f_info(self.param_cache.get_average_param())

    def f_exe(self, i_iter, w):
        """
        Place holder for now.
        """
        pass

    def _f_exe_decorated(self, i_iter, w):
        if self.param_cache_size == 1:
            return self.f_exe(i_iter, w)
        else:
            self.param_cache_size.add_param(w)
            return self.f_exe(i_iter, self.param_cache.get_average_param())

    def _prepare_for_training(self):
        self.best_obj = None
        self.best_w = None
        self.init_w = self.net.get_param_vec()

    def _process_options(self, kwargs):
        """
        Preprocess the keyword options before feeding into the optimization
        function.

        Changes will be made directly to kwargs.
        """
        pass

    def train_gradient_descent(self, **kwargs):
        """
        f_info will be overwritten here.
        """
        self._prepare_for_training()
        self.load_train_target()
        if 'f_info' not in kwargs:
            kwargs['f_info'] = self._f_info_decorated
        if 'f_exe' not in kwargs:
            kwargs['f_exe'] = self._f_exe_decorated
        self._process_options(kwargs)
        self.print_options(kwargs)
        opt.fmin_gradient_descent(self.f_and_fprime, self.init_w, **kwargs)
        return self._f_post_training_decorated()

    def train_sgd(self, **kwargs):
        """
        f_info will be overwritten here.
        """
        self._prepare_for_training()
        if 'minibatch_size' in kwargs:
            minibatch_size = kwargs['minibatch_size']
            del kwargs['minibatch_size']
        else:
            minibatch_size = 100

        self.create_minibatch_generator(minibatch_size)
        if 'f_info' not in kwargs:
            kwargs['f_info'] = self._f_info_decorated
        if 'f_exe' not in kwargs:
            kwargs['f_exe'] = self._f_exe_decorated
        self._process_options(kwargs)
        self.print_options(kwargs)
        opt.fmin_gradient_descent(self.f_and_fprime_minibatch, self.init_w, **kwargs)
        return self._f_post_training_decorated()

    def train_lbfgs(self, **kwargs):
        self._prepare_for_training()
        self.load_train_target()
        if 'weight_decay' in kwargs:
            f_and_fprime = self.get_f_and_fprime_func(weight_decay=kwargs['weight_decay'])
            del kwargs['weight_decay']
        else:
            f_and_fprime = self.f_and_fprime
        self._process_options(kwargs)
        self.print_options(kwargs)
        self.best_w, self.best_obj, _ = spopt.fmin_l_bfgs_b(f_and_fprime, self.init_w, **kwargs)
        return self.f_post_training()

    def _f_post_training_decorated(self):
        if self.param_cache_size == 1:
            return self.f_post_training()
        else:
            self.net.set_noiseless_param_from_vec(self.param_cache.get_average_param())
            return self.f_post_training()

    def f_post_training(self):
        """
        Can be customized.
        """
        self.net.set_param_from_vec(self.best_w)
        print '=============================='
        print 'Best ' + ('val' if self.use_validation else 'train') + ' obj %.4f' % self.best_obj

    def print_options(self, kwargs):
        for k, v in kwargs.iteritems():
            print '%s=%s' % (str(k), str(v))
        print ''

class ClassificationLearner(Learner):
    """
    Learner tailored to a classification problem.
    """
    def _compute_accuracy(self, t, tpred):
        return t[np.arange(len(tpred)), tpred].mean()

    def f_info(self, w):
        train_loss = None
        val_loss = None

        w_0 = self.net.get_param_vec()
        self.net.set_noiseless_param_from_vec(w)
        self.net.load_target(self.t_train)
        y = self.net.forward_prop(self.x_train, add_noise=False, compute_loss=True)
        train_loss = self.net.get_loss() / self.x_train.shape[0]
        train_acc = self._compute_accuracy(self.t_train, y.argmax(axis=1))

        if self.use_validation:
            self.net.load_target(self.t_val)
            y = self.net.forward_prop(self.x_val, add_noise=False, compute_loss=True)
            val_loss = self.net.get_loss() / self.x_val.shape[0]
            val_acc = self._compute_accuracy(self.t_val, y.argmax(axis=1))
            self.net.load_target(self.t_train)

            s = 'train loss %.4f, acc %.4f, val loss %.4f, acc ' % (train_loss, train_acc, val_loss)
            if self.best_obj is None or val_acc > self.best_obj:
                self.best_obj = val_acc 
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % val_acc)
            else:
                s += '%.4f' % val_acc
        else:
            s = 'train loss %.4f, acc ' % train_loss
            if self.best_obj is None or train_acc < self.best_obj:
                self.best_obj = train_acc
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % train_acc)
            else:
                s += '%.4f' % train_acc

        self.net.set_param_from_vec(w_0)
        return s



