import unittest

import numpy as np

from metrics import accuracy_score


class AccuracyScoreTests(unittest.TestCase):
    def test_accuracy_score_flattens_inputs(self):
        y_true = np.array([[1], [0], [1], [1]])
        y_pred = np.array([1, 0, 0, 1])

        score = accuracy_score(y_true, y_pred)

        self.assertEqual(score, 0.75)


if __name__ == "__main__":
    unittest.main()
