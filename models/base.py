from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self._add_intercept = True
        self._standardize = False
        self.method = None
        self.loss = []

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def standardize(self, X):
        return (X - self.mu) / (self.sigma + 1e-8)

    def preprocess(self, X):
        if self._standardize:
            X = self.standardize(X)
        if self._add_intercept:
            X = self.add_intercept(X)
        return X

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def visualize(self, X, y):
        pass
