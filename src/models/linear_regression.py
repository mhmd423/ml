import matplotlib.pyplot as plt
import numpy as np

from .base import Model


class LinearRegression(Model):
    METHODS = ["normal_equation", "gradient_descent"]

    def __init__(self):
        super().__init__()
        self.theta = None

    def fit(
        self,
        X,
        y,
        standardize=False,
        add_intercept=True,
        method="normal_equation",
    ):
        self._standardize = standardize
        self._add_intercept = add_intercept
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.method = method

        X_processed = self.preprocess(X)
        y = y.reshape(-1, 1)

        m, n = X_processed.shape
        self.theta = np.zeros((n, 1))

        if method == "normal_equation":
            self.theta = np.linalg.pinv(X_processed.T @ X_processed) @ X_processed.T @ y
        elif method == "gradient_descent":
            learning_rate = 0.01
            num_iterations = 1000
            eps = 1e-5

            for _ in range(num_iterations):
                hyp = X_processed @ self.theta
                gradient = (X_processed.T @ (hyp - y)) / m
                self.theta -= learning_rate * gradient

                if np.linalg.norm(gradient) < eps:
                    break
        else:
            raise ValueError(f"Method must be one of {self.METHODS}")

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

        return X_processed @ self.theta

    def visualize(self, X, y_true, more_info=False):
        m, n = X.shape
        y_true = y_true.flatten()

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

        ax = axes[0]
        ax.scatter(X, y_true, color="blue", label="Data Points", s=60, marker="o")
        ax.plot(x_plot, y_plot, color="red", label="Regression Line", linewidth=2)
        ax.set_title("Regression Fit", fontsize=13)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        if more_info:
            ax2 = axes[1]
            ax2.axis("off")
            info = (
                f"method: {self.method}\n"
                f"theta: {np.array2string(self.theta.flatten(), precision=6)}\n"
                f"standardize: {self._standardize}\n"
                f"intercept: {self._add_intercept}"
            )
            ax2.text(
                0.02,
                0.98,
                info,
                transform=ax2.transAxes,
                fontsize=11,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#dddddd"),
            )
            ax2.set_title("Model Parameters", fontsize=13)

        plt.suptitle(f"Linear Regression -- {self.method}", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return fig
