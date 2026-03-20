from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Model(ABC):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self._add_intercept = True
        self._normalize = False
        self.method = None
        self.loss = []

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def normalize(self, X):
        return (X - self.mu) / (
            self.sigma + 1e-8
        )  # Add a small epsilon to avoid division by zero

    def preprocess(self, X):
        if self._normalize:
            X = self.normalize(X)
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


class LogisticRegression(Model):
    METHODS = ["gradient_descent", "newton_method"]

    def __init__(self):
        super().__init__()
        self.theta = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def BCE_loss(hyp, y_true):
        eps = 1e-12
        hyp = np.clip(hyp, eps, 1 - eps)  # Avoid log(0)
        return -np.mean(y_true * np.log(hyp) + (1 - y_true) * np.log(1 - hyp))

    def fit(
        self,
        X,
        y,
        fit_intercept=True,
        method="gradient_descent",
        learning_rate=0.01,
        num_iterations=1000,
        normalize=False,
        eps=1e-5,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        # adding flags for use again in other methods
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self._add_intercept = fit_intercept
        self._normalize = normalize
        self.method = method
        self.loss = []

        X_processed = self.preprocess(X)

        y = y.reshape(-1, 1)  # Ensure y is a column vector
        m, n = X_processed.shape
        eps = eps

        self.theta = np.zeros((n, 1))

        if method == "gradient_descent":
            for _ in range(num_iterations):
                z = np.dot(X_processed, self.theta)
                hyp = self.sigmoid(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                self.theta -= learning_rate * gradient
                self.loss.append(self.BCE_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

        elif method == "newton_method":
            for _ in range(num_iterations):
                z = np.dot(X_processed, self.theta)
                hyp = self.sigmoid(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                H = np.dot(X_processed.T, X_processed * (hyp * (1 - hyp))) / m
                self.theta -= np.linalg.solve(H, gradient)
                self.loss.append(self.BCE_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

        return self

    def predict(
        self,
        X,
        output="binary",
    ):
        if self.theta is None:
            raise Exception("Model not trained yet")

        X_processed = self.preprocess(X)

        m, n = X_processed.shape
        if self.theta.shape[0] != n:
            raise ValueError(
                f"Expected input with {self.theta.shape[0]} features, got {n}"
            )

        if output == "probability":
            return self.sigmoid(X_processed @ self.theta)
        elif output == "binary":
            prob = self.sigmoid(X_processed @ self.theta)
            return (prob >= 0.5).astype(int)
        else:
            raise ValueError(f"Output must be 'binary' or 'probability', got {output}")

    def visualize(
        self,
        X,
        y_true,
        contour_levels=1,
    ):
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

        z = self.predict(
            grid,
            output="probability",
        ).reshape(xx1.shape)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ------------------- plot 1 -------------------
        ax1 = axes[0]
        # making the colored background using contourf
        ax1.contourf(xx1, xx2, z, levels=contour_levels, cmap="RdBu", alpha=0.6)
        ax1.contour(xx1, xx2, z, levels=[0.5], colors="k", linewidths=2)

        # plotting the original data points
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

        # -------------------- plot 2 -------------------
        ax2 = axes[1]
        if self.loss:
            ax2.plot(self.loss, color="steelblue", linewidth=2)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Binary Cross-Entropy Loss")
            ax2.set_title("Training Loss", fontsize=13)
            ax2.grid(True, linestyle="--", alpha=0.4)
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
        plt.suptitle(
            f"Logistic Regression -- {self.method}", fontsize=15, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
        return self


class LinearRegression(Model):
    METHODS = ["normal_equation", "gradient_descent",]  # one iteration of netwon method is equivalent to normal equation
    def __init__(self):
        super().__init__()
        self.theta = None

    def fit(
        self,
        X,
        y,
        normalize=False,
        add_intercept=True,
        method="normal_equation",
        ):
        
        self._normalize = normalize
        self._add_intercept = add_intercept
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.method = method
        
        X_processed = self.preprocess(X)
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        
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
    
    def visualize(self, X, y_true):
        m , n = X.shape
        y_true = y_true.flatten()  # Ensure y_true is a 1D array
        
        if n != 1:
            raise ValueError("Visualization only supported for 1D data")
        
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_plot = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        y_plot = self.predict(x_plot)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y_true, color='blue', label='Data Points', s=60, marker='o')
        plt.plot(x_plot, y_plot, color='red', label='Regression Line', linewidth=2)
        plt.title(f"Linear Regression -- {self.method}", fontsize=15, fontweight='bold')
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.show()
        
