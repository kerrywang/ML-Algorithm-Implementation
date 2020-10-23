from sklearn.datasets import load_iris
import numpy as np
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raise_message
from sklearn.utils._testing import assert_raises
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import assert_warns_message

from SupervisedLearning.LinearModel.logistic_regression import LogisticRegression
import pytest

iris = load_iris()
X = np.array([[-1, 0], [0, 1], [1, 1]])
Y1 = np.array([0, 1, 1]).reshape(X.shape[0], -1)


def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]

    clf.fit(X, y)

    # assert predicted.shape == (n_samples, 1)
    # assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    print (probabilities.shape)
    assert probabilities.shape == (n_samples, n_classes)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)

def test_2_class_prediction():
    check_predictions(LogisticRegression(), X, Y1)

if __name__ == "__main__":
    test_2_class_prediction()