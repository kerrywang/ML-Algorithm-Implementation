import numpy as np
import abc
from base import Callable, Derivable

class AbstractLoss(Callable, Derivable):
    pass

class cross_entropy_loss(AbstractLoss):
    '''
    Currently Just support binary class cross entropy
    TODO: Look up to implement multimodal cross entropy loss.
    :return:
    '''
    def __int__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return -1 * np.sum(y_true * np.log(y_pred))

    def backward(self, retval):
        '''
        dL / da = -(y * dlog(a)/da + (1 - y) * dlog(1 - a) / da
        :param retval: dL = 1
        :return: dL / d_activ =
        '''
        return -self.y_true / self.y_pred + (1 - self.y_true) / (1 - self.y_pred)

class hinge_loss(AbstractLoss):
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.sum(0, 1 - y_true * y_pred)