import numpy as np

from src.models.linear_regression import LinearRegression


def test_fits_line_with_normal_equation():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegression().fit(X, y, method="normal_equation")
    predictions = model.predict(X).reshape(-1)

    np.testing.assert_allclose(predictions, y, atol=1e-8)
