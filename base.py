import abc

class BaseEstimator(abc.ABC):
    """
    Base class for estimators, leaving it empty for now just incase i want to add more common utils later
    """
    def fit(self, **kwargs):
        pass



class Derivable(abc.ABC):
    @abc.abstractmethod
    def backward(self, *args, **kwargs):
        pass

class Callable(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class Updatable(abc.ABC):
    @abc.abstractmethod
    def update(self):
        pass