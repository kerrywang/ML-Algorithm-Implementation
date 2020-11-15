from SupervisedLearning.DecisionTree.DecisionTreeHelper import DecisionTreeBase


class DecisionTreeClassifier(DecisionTreeBase):
    '''
      Parameters
     ----------
     criterion : {"gini", "entropy"}, default="gini"
         The function to measure the quality of a split. Supported criteria are
         "gini" for the Gini impurity and "entropy" for the information gain.
     splitter : {"best", "random"}, default="best"
         The strategy used to choose the split at each node. Supported
         strategies are "best" to choose the best split and "random" to choose
         the best random split.
     max_depth : int, default=None
         The maximum depth of the tree. If None, then nodes are expanded until
         all leaves are pure or until all leaves contain less than
         min_samples_split samples.
     '''
    def __init__(self, criterion="entropy", splitter="best", max_depth=None):
        super(DecisionTreeClassifier, self).__init__(criterion, splitter, max_depth)

    def fit(self, X, y):


