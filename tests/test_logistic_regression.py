import numpy as np

from src.models.logistic_regression import LogisticRegression


def test_predicts_training_data_with_newton_method(logistic_data):
    X, y = logistic_data
    model = LogisticRegression().fit(
        X,
        y,
        method="newton_method",
        standardize=True,
        num_iterations=25,
    )

    predictions = model.predict(X).reshape(-1)
    np.testing.assert_array_equal(predictions, y)


def test_accepts_legacy_standardize_argument_name(logistic_data):
    X, y = logistic_data
    model = LogisticRegression().fit(
        X,
        y,
        method="gradient_descent",
        standarize=True,
        num_iterations=2000,
    )

    predictions = model.predict(X).reshape(-1)
    np.testing.assert_array_equal(predictions, y)
