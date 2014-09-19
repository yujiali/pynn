"""
Definition of losses in a neural network.

Yujia Li, 09/2014
"""

import gnumpy as gnp

class Loss(object):
    """
    Base class for losses.
    """
    def __init__(self):
        self.loss_value = 0

    def load_target(self, target, *args, **kwargs):
        """
        Load targets that will be used in loss and loss gradient computation.
        """
        pass

    def compute_loss_and_grad(self, pred, compute_grad=False):
        """
        Compute loss and output gradient.

        pred: (n_cases, n_dims) prediction matrix
        target: (n_cases, n_dims) target matrix, or anything else that the
            particular loss needs.
        compute_grad: compute prediction gradient if True

        Return: (loss, grad)
            loss is one single number, the loss for the predictions
            grad: (n_cases, n_dims) gradient matrix, if compute_grad set to
                False, this may not be computed to save time.
        """
        raise NotImplementedError()

    def get_most_recent_loss(self):
        """
        Return the most recent loss computed in compute_loss_and_grad.
        """
        return self.loss_value

    def get_name(self):
        raise NotImplementedError()

    def target_should_be_normalized(self):
        """
        Mainly for testing.
        
        Property of the loss that for each row the target matrix should be 
        positive and sum to one.
        """
        return False

    def target_should_be_one_hot(self):
        """
        Mainly for testing.

        Property of the loss that for each row the target matrix should only
        have one entry to be one and all others be zero.
        """
        return False

class LossManager(object):
    """
    This maintains a set of different losses.
    """
    def __init__(self):
        self.loss_dict = {}

    def register_loss(self, loss):
        loss_name = loss.get_name()
        self.loss_dict[loss_name] = loss
        globals()['LOSS_NAME_' + loss_name.upper()] = loss_name

    def get_loss_list(self):
        return self.loss_dict.values()

    def get_loss_instance(self, loss_type):
        return self.loss_dict[loss_type]

_loss_manager = LossManager()

def register_loss(loss):
    _loss_manager.register_loss(loss)

def get_loss_from_type_name(loss_type):
    return _loss_manager.get_loss_instance(loss_type)

class DebugLoss(Loss):
    """
    Used for debugging only.
    """
    def compute_loss_and_grad(self, pred, compute_grad=False):
        self.loss_value = pred.sum()
        return self.loss_value, gnp.ones(pred.shape)

    def get_name(self):
        return 'debug'

register_loss(DebugLoss())

class ZeroLoss(Loss):
    """
    Simply no loss.
    """
    def compute_loss_and_grad(self, pred, compute_grad=False):
        self.loss_value = 0
        return self.loss_value, gnp.zeros(pred.shape)

    def get_name(self):
        return 'zero'

register_loss(ZeroLoss())

class SquaredLoss(Loss):
    """
    Squared loss (x - t)**2
    """
    def load_target(self, target, *args, **kwargs):
        self.target = target

    def compute_loss_and_grad(self, pred, compute_grad=False):
        diff = pred - self.target
        self.loss_value = (diff**2).sum() / 2
        return self.loss_value, diff

    def get_name(self):
        return 'squared'

register_loss(SquaredLoss())

class CrossEntropy(Loss):
    """
    Cross entropy loss -sum_ij t_ij log(y_ij) where
        y_ij = exp(x_ij) / sum_j exp(x_ij),
    so y_i must be a probability distribution, and t_i has a one-hot
    representation.
    """
    def load_target(self, target, *args, **kwargs):
        self.target = target

    def compute_loss_and_grad(self, pred, compute_grad=False):
        y = gnp.exp(pred - pred.max(axis=1)[:,gnp.newaxis])
        y = y / y.sum(axis=1)[:,gnp.newaxis]

        self.loss_value = -(self.target * gnp.log(y)).sum()
        return self.loss_value, y - self.target 

    def get_name(self):
        return 'crossentropy'

    def target_should_be_one_hot(self):
        return True

register_loss(CrossEntropy())

LOSS_LIST = _loss_manager.get_loss_list()

