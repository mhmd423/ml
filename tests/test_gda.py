import numpy as np
import pytest

from src.models.gda import GDA


def test_calculate_parameters_returns_expected_shapes(gda_data):
    X, y = gda_data
    phi, mu_0, mu_1, cov, cov_inv = GDA.calculate_parameters(X, y)

    assert phi == pytest.approx(0.5)
    assert mu_0.shape == (2,)
    assert mu_1.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov_inv.shape == (2, 2)


def test_predicts_training_data(gda_data):
    X, y = gda_data
    model = GDA().fit(X, y, standardize=True)

    predictions = model.predict(X).reshape(-1)
    np.testing.assert_array_equal(predictions, y)


def test_accepts_legacy_calculate_parameters_name(gda_data):
    X, y = gda_data
    new_result = GDA.calculate_parameters(X, y)
    legacy_result = GDA.calcluate_paramaters(X, y)

    for new_value, legacy_value in zip(new_result, legacy_result):
        np.testing.assert_allclose(new_value, legacy_value)
