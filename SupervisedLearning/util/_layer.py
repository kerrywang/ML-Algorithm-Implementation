from base import Derivable, Callable, Updatable
import numpy as np


class AbstractLayer(Derivable, Callable, Updatable):
    pass

class LinearLayer(AbstractLayer):
    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = np.random.uniform((input_dims, output_dims)).reshape((input_dims, output_dims))
        self.bias = np.zeros((output_dims, 1))
        self.prev_input = None

    def __call__(self, X):
        self.prev_input = X
        return X.dot(self.weight) + self.bias

    def backward(self, retval):
        # update the weight and bias
        self.weight_grad = self.prev_input.T.dot(retval)
        self.bias_grad = np.sum(np.ones((1, self.output_dims)) * retval)

        # return the propagated input to the prev layer
        return retval.dot(self.weight.T)

    def update(self, lr):
        self.weight -= self.weight_grad * lr
        self.bias -= self.bias_grad * lr

