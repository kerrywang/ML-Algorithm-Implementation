from base import BaseEstimator


class DecisionTreeBase(BaseEstimator):
    def __init__(self, criterion, splitter, max_depth):
        self.max_depth = max_depth


