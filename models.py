from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def visualize(self, X, y):
        pass


class LogisticRegression(Model):
    methods = ["gradient_descent", "newton_method", "stochastic_gradient_descent"]

    def __init__(self):
        self.theta = None

    def fit(
        self,
        X,
        y,
        fit_intercept=True,
        method="gradient_descent",
        learning_rate=0.01,
        num_iterations=1000,
    ):
        if method not in self.methods:
            raise ValueError(f"Method must be one of {self.methods}")

        if fit_intercept:
            X = self.add_intercept(X)

        y = y.flatten()  # Ensure y is a 1D array
        m, n = X.shape
        eps = 1e-5

        self.theta = np.zeros(n)

        if method == "gradient_descent":
            for _ in range(num_iterations):
                z = np.dot(X, self.theta)
                hyp = 1 / (1 + np.exp(-z))
                gradient = np.dot(X.T, (hyp - y)) / m
                self.theta -= learning_rate * gradient

                if np.linalg.norm(gradient) < eps:
                    break

        elif method == "newton_method":
            for _ in range(num_iterations):
                z = np.dot(X, self.theta)
                hyp = 1 / (1 + np.exp(-z))
                gradient = np.dot(X.T, (hyp - y)) / m
                H = np.dot(X.T, X * (hyp * (1 - hyp))) / m
                self.theta -= np.linalg.solve(H, gradient)

                if np.linalg.norm(gradient) < eps:
                    break

        elif method == "stochastic_gradient_descent":
            for _ in range(num_iterations):
                for i in range(m):
                    z = np.dot(X[i], self.theta)
                    hyp = 1 / (1 + np.exp(-z))
                    gradient = X[i] * (hyp - y[i])
                    self.theta -= learning_rate * gradient

                if np.linalg.norm(gradient) < eps:
                    break

    def predict(self, X):
        pass

    def visualize(self, X, y):
        pass
