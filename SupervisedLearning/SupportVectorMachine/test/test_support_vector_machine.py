from SupervisedLearning.SupportVectorMachine.support_vector_machine import SupportVectorMachine
import numpy as np
from sklearn.datasets import load_iris

def get_non_linearly_separable_dataset():
    # 1d dataset (Obviously not linearly seperable)
    X = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    return X, y

def get_linearly_separable_dataset():
    data = load_iris()
