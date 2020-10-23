import numpy as np
import abc
from base import Derivable, Callable

class abstract_activation(Derivable, Callable):
    pass

class sigmoid(abstract_activation):
    '''
    Sigmoid normalize the output between 0 and 1
    http://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
    :param X:
    :return:
    '''
    def __init__(self):
        self.output = None

    def __call__(self, y):
        self.output = 1 / (1 + np.exp(-y))
        return self.output

    def backward(self, retval):
        '''
        derivative of sigmoid = a * (1 - a) where a is the output
        :param retval: dL / dsigmoid
        :return: dL / dy  <= dL / d_sig * d_sig / dy
        '''
        return self.output * (1 - self.output) * retval

class soft_max(abstract_activation):
    '''
    Softmax is a activation function that ensures probabilities of all classes sums to 1
    In a 2 class situation, softmax is the same as sigmoid function. See https://stats.stackexchange.com/questions/87248/is-binary-logistic-regression-a-special-case-of-multinomial-logistic-regression#comment609940_87270

    P(y_i|x, W) = exp(W_i*x (yi) ) / SUM(exp(W_k *x (y_k) ) for k in range(num_classes))
    :param x:
    :return:
    '''

    def __call__(self, y):
        e_x = np.exp(y - np.max(y, keepdims=True)) # subtracting max here for numeric stability
        return e_x / np.sum(e_x, axis=0)


    def backward(self, retval):
        '''
        Derivative of softmax can be found here: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        :return:
        '''
