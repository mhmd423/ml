import numpy as np
import pytest


@pytest.fixture
def logistic_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [1.0, 1.5],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def gda_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -1.5],
            [1.0, 1.0],
            [2.0, 1.5],
        ]
    )
    y = np.array([0, 0, 1, 1])
    return X, y
