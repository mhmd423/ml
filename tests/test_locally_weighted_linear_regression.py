import numpy as np

from src.models.locally_weighted_linear_regression import lw_LinearRegression


def test_predict_matches_training_targets_with_small_bandwidth():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = lw_LinearRegression(tau=0.3, lamda=1e-8).fit(X, y)
    predictions = model.predict(X).reshape(-1)

    np.testing.assert_allclose(predictions, y, atol=1e-3)


def test_predict_returns_value_per_query_point():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = lw_LinearRegression(tau=0.5).fit(X, y)
    predictions = model.predict(np.array([[1.5], [2.5]]))

    assert predictions.shape == (2, 1)
