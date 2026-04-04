import unittest

import numpy as np

from models import GDA, LinearRegression, LogisticRegression


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [1.0, 1.5],
                [2.0, 1.0],
            ]
        )
        self.y = np.array([0, 0, 1, 1])

    def test_predicts_training_data_with_newton_method(self):
        model = LogisticRegression().fit(
            self.X,
            self.y,
            method="newton_method",
            standardize=True,
            num_iterations=25,
        )

        predictions = model.predict(self.X).reshape(-1)

        np.testing.assert_array_equal(predictions, self.y)

    def test_accepts_legacy_standardize_argument_name(self):
        model = LogisticRegression().fit(
            self.X,
            self.y,
            method="gradient_descent",
            standarize=True,
            num_iterations=2000,
        )

        predictions = model.predict(self.X).reshape(-1)

        np.testing.assert_array_equal(predictions, self.y)


class LinearRegressionTests(unittest.TestCase):
    def test_fits_line_with_normal_equation(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 3.0, 5.0, 7.0])

        model = LinearRegression().fit(X, y, method="normal_equation")
        predictions = model.predict(X).reshape(-1)

        np.testing.assert_allclose(predictions, y, atol=1e-8)


class GDATests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -1.5],
                [1.0, 1.0],
                [2.0, 1.5],
            ]
        )
        self.y = np.array([0, 0, 1, 1])

    def test_calculate_parameters_returns_expected_shapes(self):
        phi, mu_0, mu_1, cov, cov_inv = GDA.calculate_parameters(self.X, self.y)

        self.assertAlmostEqual(phi, 0.5)
        self.assertEqual(mu_0.shape, (2,))
        self.assertEqual(mu_1.shape, (2,))
        self.assertEqual(cov.shape, (2, 2))
        self.assertEqual(cov_inv.shape, (2, 2))

    def test_predicts_training_data(self):
        model = GDA().fit(self.X, self.y, standardize=True)

        predictions = model.predict(self.X).reshape(-1)

        np.testing.assert_array_equal(predictions, self.y)

    def test_accepts_legacy_calculate_parameters_name(self):
        new_result = GDA.calculate_parameters(self.X, self.y)
        legacy_result = GDA.calcluate_paramaters(self.X, self.y)

        for new_value, legacy_value in zip(new_result, legacy_result):
            np.testing.assert_allclose(new_value, legacy_value)


if __name__ == "__main__":
    unittest.main()
