import numpy as np
import pytest

from src.models.poisson_regression import PoissonRegression


def test_predict_returns_positive_rates():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([1.0, 1.0, 2.0, 3.0])

    model = PoissonRegression().fit(
        X,
        y,
        method="newton_method",
        standardize=True,
        num_iterations=100,
    )
    predictions = model.predict(X)

    assert predictions.shape == (4, 1)
    assert np.all(predictions > 0)


def test_rejects_negative_targets():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, -1.0, 2.0])

    with pytest.raises(ValueError, match="non-negative"):
        PoissonRegression().fit(X, y)
