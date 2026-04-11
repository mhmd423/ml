import matplotlib.pyplot as plt
import numpy as np

from .base import Model


class GDA(Model):
    def __init__(self):
        super().__init__()
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.cov = None
        self.cov_inv = None
        self.theta = None
        self.theta_0 = None

    @staticmethod
    def calculate_parameters(X, y):
        m, n = X.shape
        X0 = X[y == 0]
        X1 = X[y == 1]
        phi = np.mean(y)
        mu_0 = X0.mean(axis=0)
        mu_1 = X1.mean(axis=0)

        cov = ((X0 - mu_0).T @ (X0 - mu_0) + (X1 - mu_1).T @ (X1 - mu_1)) / m

        cov_inv = np.linalg.inv(cov)
        return phi, mu_0, mu_1, cov, cov_inv

    @staticmethod
    def calcluate_paramaters(X, y):
        return GDA.calculate_parameters(X, y)

    def fit(self, X, y, standardize=False):
        y = y.flatten()

        self._standardize = standardize
        self._add_intercept = False
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        X_processed = self.preprocess(X)

        self.phi, self.mu_0, self.mu_1, self.cov, self.cov_inv = self.calculate_parameters(
            X_processed, y
        )

        self.theta = self.cov_inv @ (self.mu_1 - self.mu_0)

        self.theta_0 = (
            -0.5 * self.mu_1.T @ self.cov_inv @ self.mu_1
            + 0.5 * self.mu_0.T @ self.cov_inv @ self.mu_0
            + np.log(self.phi / (1 - self.phi))
        )
        return self

    def predict(self, X, output="binary"):
        if self.phi is None:
            raise Exception("Model not trained yet")

        X_processed = self.preprocess(X)

        z = X_processed @ self.theta + self.theta_0

        if output == "probability":
            return 1 / (1 + np.exp(-z))
        else:
            return (z >= 0).astype(int)

    def visualize(self, X, y_true, contour_levels=1, more_info=False):
        m, n = X.shape
        y_true = y_true.flatten()
        if n != 2:
            raise ValueError("Visualization only supported for 2D data")

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
        )
        grid = np.c_[xx1.ravel(), xx2.ravel()]

        z = self.predict(grid, output="probability").reshape(xx1.shape)

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
        ax1.contourf(xx1, xx2, z, levels=contour_levels, cmap="RdBu", alpha=0.6)
        ax1.contour(xx1, xx2, z, levels=[0.5], colors="k", linewidths=2)

        x_true = X[y_true == 1]
        ax1.scatter(
            x_true[:, 0],
            x_true[:, 1],
            c="b",
            label="class 1",
            s=60,
            marker="o",
            zorder=3,
        )

        x_false = X[y_true == 0]
        ax1.scatter(
            x_false[:, 0],
            x_false[:, 1],
            c="r",
            label="class 0",
            s=60,
            marker="x",
            zorder=3,
        )

        ax1.set_title("Decision Boundary", fontsize=13)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.4)

        if more_info:
            ax2 = axes[1]
            ax2.axis("off")
            sigma_det = np.linalg.det(self.cov) if self.cov is not None else np.nan
            model_info = (
                f"phi: {self.phi:.4f}\n"
                f"mu_0: {np.array2string(self.mu_0, precision=4)}\n"
                f"mu_1: {np.array2string(self.mu_1, precision=4)}\n"
                f"det(sigma): {sigma_det:.6f}\n"
                f"theta_0: {self.theta_0:.4f}\n"
                f"theta: {np.array2string(self.theta, precision=4)}"
            )
            ax2.text(
                0.02,
                0.98,
                model_info,
                transform=ax2.transAxes,
                fontsize=11,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#dddddd"),
            )
            ax2.set_title("Model Parameters", fontsize=13)

        plt.suptitle("Gaussian Discriminant Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()

        return fig
