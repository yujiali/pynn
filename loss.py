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
        pass

    def load_target(self, target):
        """
        Load targets that will be used in loss and loss gradient computation.
        """
        raise NotImplementedError()

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

class _DebugLoss(Loss):
    """
    Used for debugging only.
    """
    def compute_loss_and_grad(self, pred, compute_grad=False):
        return pred.sum(), gnp.ones(pred.shape, dtype=pred.dtype)


