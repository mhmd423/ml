import matplotlib.pyplot as plt
import numpy as np

from .base import Model


class PoissonRegression(Model):
    METHODS = ["gradient_descent", "newton_method"]

    def __init__(self):
        super().__init__()
        self.theta = None
        self.lambda_ = None

    @staticmethod
    def poisson_loss(hyp, y_true):
        eps = 1e-12
        hyp = np.clip(hyp, eps, None)
        return np.mean(hyp - y_true * np.log(hyp))

    def fit(
        self,
        X,
        y,
        fit_intercept=True,
        method="newton_method",
        learning_rate=0.01,
        num_iterations=1000,
        standardize=False,
        eps=1e-5,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self._add_intercept = fit_intercept
        self._standardize = standardize
        self.method = method
        self.loss = []

        X_processed = self.preprocess(X)

        y = y.reshape(-1, 1)
        if np.any(y < 0):
            raise ValueError("Poisson regression requires non-negative targets")
        if not np.all(np.isfinite(y)):
            raise ValueError("Targets contain NaN or infinite values")
        m, n = X_processed.shape

        self.theta = np.zeros((n, 1))
        ridge = 1e-8 * np.eye(n)

        if method == "gradient_descent":
            for _ in range(num_iterations):
                z = np.dot(X_processed, self.theta)
                z = np.clip(z, -30, 30)
                hyp = np.exp(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                self.theta -= learning_rate * gradient
                self.loss.append(self.poisson_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

        elif method == "newton_method":
            for _ in range(num_iterations):
                z = np.dot(X_processed, self.theta)
                z = np.clip(z, -30, 30)
                hyp = np.exp(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                H = np.dot(X_processed.T, X_processed * hyp) / m + ridge
                self.theta -= np.linalg.solve(H, gradient)
                self.loss.append(self.poisson_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

        self.lambda_ = np.exp(np.clip(X_processed @ self.theta, -30, 30))
        return self

    def predict(self, X):
        if self.theta is None:
            raise Exception("Model not trained yet")

        X_processed = self.preprocess(X)

        m, n = X_processed.shape
        if self.theta.shape[0] != n:
            raise ValueError(
                f"Expected input with {self.theta.shape[0]} features, got {n}"
            )

        return np.exp(np.clip(X_processed @ self.theta, -30, 30))

    def visualize(self, X, y, more_info=False):
        m, n = X.shape
        y = y.flatten()
        if n != 1:
            raise ValueError("Visualization only supported for 1D data")

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_plot = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        y_plot = self.predict(x_plot)

        if more_info:
            width = 14
            count = 2
        else:
            width = 7
            count = 1
        fig, axes = plt.subplots(1, count, figsize=(width, 6))

        if count == 1:
            axes = [axes]

        ax1 = axes[0]
        ax1.scatter(X, y, color="blue", label="Data Points", s=60, marker="o")
        ax1.plot(x_plot, y_plot, color="red", label="Poisson Mean", linewidth=2)
        ax1.set_title("Poisson Fit", fontsize=13)
        ax1.set_xlabel("Feature")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)

        if more_info:
            ax2 = axes[1]
            if self.loss:
                ax2.plot(self.loss, color="steelblue", linewidth=2)
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Poisson Loss")
                ax2.set_title("Training Loss", fontsize=13)
                ax2.grid(True, linestyle="--", alpha=0.4)
                ax2.text(
                    0.95,
                    0.95,
                    f"last-error: {self.loss[-1]}",
                    transform=ax2.transAxes,
                    fontsize=10,
                    color="white",
                    ha="right",
                    va="top",
                    bbox=dict(facecolor="black", alpha=1, edgecolor="none"),
                )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No loss history",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    fontsize=12,
                    color="grey",
                )

        plt.suptitle(
            f"Poisson Regression -- {self.method}", fontsize=15, fontweight="bold"
        )
        plt.tight_layout()
        return fig
