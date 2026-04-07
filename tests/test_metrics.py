import numpy as np

from metrics import accuracy_score


def test_accuracy_score_flattens_inputs():
    y_true = np.array([[1], [0], [1], [1]])
    y_pred = np.array([1, 0, 0, 1])

    score = accuracy_score(y_true, y_pred)

    assert score == 0.75
