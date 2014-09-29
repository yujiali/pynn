"""
Definition of losses in a neural network.

Yujia Li, 09/2014
"""

import gnumpy as gnp

class Loss(object):
    """
    Base class for losses.
    """
    def __init__(self, **kwargs):
        """
        kwargs is allowed for passing in extra information needed for initialization.
        """
        self.loss_value = 0
        self.weight = kwargs.get('weight', 1)

    def load_target(self, target, *args, **kwargs):
        """
        Load targets that will be used in loss and loss gradient computation.
        """
        pass

    def set_weight(self, weight):
        """
        Specify the weight for this loss, which is a constant scale factor
        that both loss and gradient will use.
        """
        self.weight = weight

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        """
        Compute loss and output gradient.

        pred: (n_cases, n_dims) prediction matrix
        target: (n_cases, n_dims) target matrix, or anything else that the
            particular loss needs.
        compute_grad: compute prediction gradient if True

        Return: (loss, grad)
            loss is one single number, the loss for the predictions
            grad: (n_cases, n_dims) gradient matrix, if compute_grad set to
                False, this may not be computed to save time, but a place
                holder will be used.
        """
        raise NotImplementedError()

    def compute_loss_and_grad(self, pred, compute_grad=False):
        """
        Compute the weighted loss and loss gradient.

        See compute_not_weighted_loss_and_grad for details.

        User of the Loss object should use this function.
        """
        loss, loss_grad = self.compute_not_weighted_loss_and_grad(pred, compute_grad)
        if self.weight == 1:
            self.loss_value = loss
            return self.loss_value, loss_grad
        else:
            self.loss_value = loss * self.weight
            return self.loss_value, loss_grad * self.weight

    def get_most_recent_loss(self):
        """
        Return the most recent loss computed in compute_loss_and_grad.
        """
        return self.loss_value

    def get_name(self):
        raise NotImplementedError()

    def get_id(self):
        """
        Loss type ID, 0 reserved for no loss.
        """
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

    def __repr__(self):
        return 'Loss <%s> w=%g' % (self.get_name(), self.weight)

_LOSS_ID_NONE = 0

class LossManager(object):
    """
    This maintains a set of different losses.
    """
    def __init__(self):
        self.name_to_loss = {}
        self.id_to_loss = {}

    def register_loss(self, loss):
        loss_name = loss.get_name()
        loss_id = loss.get_id()
        self.name_to_loss[loss_name] = loss.__class__
        self.id_to_loss[loss_id] = loss.__class__
        globals()['LOSS_NAME_' + loss_name.upper()] = loss_name
        globals()['_LOSS_ID_' + loss_name.upper()] = loss_id

    def get_loss_list(self):
        return [self.get_loss_instance(loss_type) \
                for loss_type in self.name_to_loss.keys()]

    def get_loss_instance(self, loss_type, **kwargs):
        return self.name_to_loss[loss_type](**kwargs)

    def get_loss_instance_from_id(self, loss_id, **kwargs):
        if loss_id == _LOSS_ID_NONE:
            return None
        return self.id_to_loss[loss_id](**kwargs)

_loss_manager = LossManager()

def register_loss(loss):
    _loss_manager.register_loss(loss)

def get_loss_from_type_name(loss_type, **kwargs):
    return _loss_manager.get_loss_instance(loss_type, **kwargs)

def get_loss_from_type_id(loss_id, **kwargs):
    return _loss_manager.get_loss_instance_from_id(loss_id, **kwargs)

class DebugLoss(Loss):
    """
    Used for debugging only.
    """
    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        return pred.sum(), gnp.ones(pred.shape)

    def get_name(self):
        return 'debug'

    def get_id(self):
        return 1

register_loss(DebugLoss())

class ZeroLoss(Loss):
    """
    Simply no loss.
    """
    def compute_loss_and_grad(self, pred, compute_grad=False):
        return 0, gnp.zeros(pred.shape)

    def get_name(self):
        return 'zero'

    def get_id(self):
        return 2

register_loss(ZeroLoss())

class SquaredLoss(Loss):
    """
    Squared loss (x - t)**2
    """
    def load_target(self, target, *args, **kwargs):
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        diff = pred - self.target
        return (diff**2).sum() / 2, diff

    def get_name(self):
        return 'squared'

    def get_id(self):
        return 3

register_loss(SquaredLoss())

class CrossEntropy(Loss):
    """
    Cross entropy loss -sum_ij t_ij log(y_ij) where
        y_ij = exp(x_ij) / sum_j exp(x_ij),
    so y_i must be a probability distribution, and t_i has a one-hot
    representation.
    """
    def load_target(self, target, *args, **kwargs):
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        y = gnp.exp(pred - pred.max(axis=1)[:,gnp.newaxis])
        y = y / y.sum(axis=1)[:,gnp.newaxis]

        return -(self.target * gnp.log(y)).sum(), y - self.target

    def get_name(self):
        return 'crossentropy'

    def target_should_be_one_hot(self):
        return True

    def get_id(self):
        return 4

register_loss(CrossEntropy())

LOSS_LIST = _loss_manager.get_loss_list()

