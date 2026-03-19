from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Model(ABC):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.add_intercept = True
        self.normalize = False
        self.loss = []
        
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def normalize(self, X):
        return (X - self.mu) / (self.sigma + 1e-8)  # Add a small epsilon to avoid division by zero
    
    def preprocess(self, X):  
        if self.normalize:
            X = self.normalize(X)
        if self.add_intercept:
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
    ):
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        # adding flags for use again in other methods
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.add_intercept = fit_intercept
        self.normalize = normalize
        
        X_processed = self.preprocess(X)
        

        y = y.reshape(-1, 1)  # Ensure y is a column vector
        m, n = X.shape
        eps = 1e-5

        self.theta = np.zeros((n, 1))

        if method == "gradient_descent":
            for _ in range(num_iterations):
                z = np.dot(X, self.theta)
                hyp = self.sigmoid(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                self.theta -= learning_rate * gradient
                self.loss.append(self.BCE_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

        elif method == "newton_method":
            for _ in range(num_iterations):
                z = np.dot(X, self.theta)
                hyp = self.sigmoid(z)
                gradient = np.dot(X_processed.T, (hyp - y)) / m
                H = np.dot(X_processed.T, X_processed * (hyp * (1 - hyp))) / m
                self.theta -= np.linalg.solve(H, gradient)
                self.loss.append(self.BCE_loss(hyp, y))

                if np.linalg.norm(gradient) < eps:
                    break

    def predict(self, X, output="binary",):
        if self.theta is None:
            raise Exception("Model not trained yet")
        
        X_processed = self.preprocess(X)

        m, n = X_processed.shape
        if self.theta.shape[0] != n:
            raise ValueError(
                f"Expected input with {self.theta.shape[0]} features, got {n}"
            )

        if output == "probability":
            return 1 / (1 + np.exp(-X_processed @ self.theta))
        elif output == "binary":
            prob = 1 / (1 + np.exp(-X_processed @ self.theta))
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
