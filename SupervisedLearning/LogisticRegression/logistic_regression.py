import SupervisedLearning.util._activation_function as activation_functions
import SupervisedLearning.util.Loss._loss_function as loss_functions
from SupervisedLearning.util._layer import LinearLayer

from base import BaseEstimator

class LogisticRegression(BaseEstimator):
    '''
    Binominal Logsitic Regression. #TODO make it support multimodal logsitic regression
    '''
    def __init__(self, activation="sigmoid", loss="cross_entropy_loss", learning_rate=0.01, solve="back_propagation"):
        self.activation = getattr(activation_functions, activation)()
        self.loss_func = getattr(loss_functions, loss)()
        self.linear_layer = None
        self.lr = learning_rate

    def fit(self, X, y, iteration=10000):
        in_dim, out_dim = X.shape[1], y.shape[1]

        self.linear_layer = LinearLayer(input_dims=in_dim, output_dims=out_dim)
        for cur_iter in range(iteration):
            # forward pass
            out = self.linear_layer(X)
            act = self.activation(out)
            loss = self.loss_func(act, y)

            loss_back = self.loss_func.backward(1)
            act_back = self.activation.backward(loss_back)
            self.linear_layer.backward(act_back)

            self.linear_layer.update(self.lr)

            if cur_iter % 10 == 0:
                print("Iteration: {}  Current Loss: {}".format(cur_iter, loss))
        return self

    def predict_proba(self, X):
        if not self.linear_layer:
            raise ValueError("Model is not fitted")

        return self.activation(self.linear_layer(X))

