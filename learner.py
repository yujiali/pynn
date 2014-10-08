"""
Learners updates the model and handles data I/O.

Yujia Li, 09/2014
"""
import pyopt.opt as opt
import color as co
import numpy as np
import scipy.optimize as spopt

class Learner(object):
    """
    Base class for all learners.
    """
    def __init__(self, net):
        """
        net is a BaseNeuralNet instance.
        """
        self.net = net

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

    def f_exe(self, w):
        """
        Place holder for now.
        """
        pass

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
        kwargs['f_info'] = self.f_info
        self._process_options(kwargs)
        self.print_options(kwargs)
        opt.fmin_gradient_descent(self.f_and_fprime, self.init_w, **kwargs)
        self.f_post_training()

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
        self.f_post_training()

    def train_sgd(self, *args, **kwargs):
        pass

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



