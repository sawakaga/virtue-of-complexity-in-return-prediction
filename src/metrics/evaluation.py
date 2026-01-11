import numpy as np
from numpy.typing import NDArray


def calculate_mse(y_true: NDArray, y_pred: NDArray):
    """
    Calculate Mean Squared Error.
    """

    return np.mean((y_true - y_pred) ** 2)


def calculate_r2(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Calculate R² (coefficient of determination).

    R² represents the proportion of variance in the dependent variable
    that is predictable from the independent variables.
    """

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
