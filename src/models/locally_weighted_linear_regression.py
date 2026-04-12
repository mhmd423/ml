import matplotlib.pyplot as plt
import numpy as np

from .base import Model


class LocallyWeightedLinearRegression(Model):
    def __init__(self, tau=0.1, lamda=1e-5):
        super().__init__()
        self.theta = None
        self.tau = tau
        self.lamda = lamda
        self.X_train = None
        self.y_train = None

    def fit(self, X, y, tau=None, fit_intercept=True, standardize=False, lamda=None):
        if tau is not None:
            self.tau = tau
        if lamda is not None:
            self.lamda = lamda

        if self.tau is None or self.tau <= 0:
            raise ValueError("tau must be a positive number")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        self._add_intercept = fit_intercept
        self._standardize = standardize
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)

        self.X_train = self.preprocess(X)
        self.y_train = y

        return self

    def predict(self, X, tau=None, lamda=None):
        if self.X_train is None or self.y_train is None:
            raise Exception("Model not trained yet")

        if tau is not None:
            self.tau = tau
        if lamda is not None:
            self.lamda = lamda

        X = np.asarray(X, dtype=float)
        X_processed = self.preprocess(X)
        m, n = X_processed.shape

        if self.X_train.shape[1] != n:
            raise ValueError(
                f"Expected input with {self.X_train.shape[1]} features, got {n}"
            )

        y_pred = np.zeros((m, 1))
        reg = self.lamda * np.eye(n)
        if self._add_intercept:
            reg[0, 0] = 0.0

        for i, x_query in enumerate(X_processed):
            diff = self.X_train - x_query
            sq_dist = np.sum(diff * diff, axis=1)

            # Shift exponents to keep the kernel computation numerically stable.
            scaled_dist = -sq_dist / (2 * self.tau**2)
            scaled_dist -= np.max(scaled_dist)
            weights = np.exp(scaled_dist)

            weighted_X = self.X_train * weights[:, None]
            weighted_y = self.y_train * weights[:, None]

            xtwx = self.X_train.T @ weighted_X
            xtwy = self.X_train.T @ weighted_y

            try:
                theta = np.linalg.solve(xtwx + reg, xtwy)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(xtwx + reg) @ xtwy

            y_pred[i, 0] = (x_query @ theta).item()

        return y_pred

    def visualize(self, X, y):
        if self.X_train is None or self.y_train is None:
            raise Exception("Model not trained yet")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.shape[1] != 1:
            raise ValueError("Visualization only supported for 1D features")

        x_min, x_max = X.min() - 1, X.max() + 1
        x_vis = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_vis = self.predict(x_vis)

        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color="blue", label="Training Data")
        plt.plot(x_vis, y_vis, color="red", label="LWLR Fit")
        plt.title(
            f"Locally Weighted Linear Regression -- tau={self.tau}, lambda={self.lamda}"
        )
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.grid()
        plt.show()


# Backward-compatible alias.
lw_LinearRegression = LocallyWeightedLinearRegression
