from sklearn.base import BaseEstimator, TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, index):
        self.index = index

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, self.index]
