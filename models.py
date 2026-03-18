from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Model(ABC):
    def __init__(self):
        self.mu = None
        self.sigma = None
        
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def normalize(self, X):
        return (X - self.mu) / self.sigma

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
    methods = ["gradient_descent", "newton_method"]

    def __init__(self):
        super().__init__()
        self.theta = None

    def fit(
        self,
        X,
        y,
        fit_intercept=True,
        method="gradient_descent",
        learning_rate=0.01,
        num_iterations=1000,
        normalize=False,
    ):
        if method not in self.methods:
            raise ValueError(f"Method must be one of {self.methods}")

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        if normalize:
            X = self.normalize(X)
            
        if fit_intercept:
            X = self.add_intercept(X)

        y = y.reshape(-1, 1)  # Ensure y is a column vector
        m, n = X.shape
        eps = 1e-5

        self.theta = np.zeros((n, 1))

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

    def predict(self, X, has_intercept=False, output="binary", normalize=False):
        if self.theta is None:
            raise Exception("Model not trained yet")
        
        if normalize:
            X = self.normalize(X)

        if not has_intercept:
            X = self.add_intercept(X)

        m, n = X.shape
        if self.theta.shape[0] != n:
            raise ValueError(
                f"Expected input with {self.theta.shape[0]} features, got {n}"
            )

        if output == "probability":
            return 1 / (1 + np.exp(-X @ self.theta))
        elif output == "binary":
            prob = 1 / (1 + np.exp(-X @ self.theta))
            return (prob >= 0.5).astype(int)
        else:
            raise ValueError(f"Output must be 'binary' or 'probability', got {output}")

    def visualize(self, X, y_true, levels=1,normalize=False):
        m, n = X.shape
        y_true = y_true.flatten()  # Ensure y_true is a 1D array
        if n != 2:
            raise ValueError("Visualization only supported for 2D data")

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
        )
        grid = np.c_[xx1.ravel(), xx2.ravel()]

        z = self.predict(grid, output="probability", normalize=normalize).reshape(xx1.shape)

        fig, ax = plt.subplots(figsize=(8, 8))

        # making the colored background using contourf
        ax.contourf(xx1, xx2, z, levels=levels, cmap="RdBu", alpha=0.6)
        ax.contour(xx1, xx2, z, levels=[0.5], colors="k", linewidths=2)

        # plotting the original data points
        x_true = X[y_true == 1]
        ax.scatter(
            x_true[:, 0],
            x_true[:, 1],
            c="b",
            label="class 1",
            s=60,
            marker="o",
            zorder=3,
        )

        x_false = X[y_true == 0]
        ax.scatter(
            x_false[:, 0],
            x_false[:, 1],
            c="r",
            label="class 0",
            s=60,
            marker="x",
            zorder=3,
        )

        ax.set_title("Logistic Regression")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
