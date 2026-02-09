import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Ridge


def linear_regression(X: NDArray, y: NDArray) -> NDArray:
    """
    Ordinary Least Squares (OLS) linear regression using normal equations.
    Computes: beta_hat = (X^T X)^(-1) X^T y
    """

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat


def ridgeless_regression(X: NDArray, y: NDArray) -> NDArray:
    """
    Ridgeless regression using minimum-norm least squares solution.

    Computes: beta_hat = X^T (X X^T)^(-1) y

    This is equivalent to the Moore-Penrose pseudoinverse solution
    and finds the minimum L2-norm solution among all solutions that
    perfectly interpolate the training data.
    """

    beta_hat = X.T @ np.linalg.inv(X @ X.T) @ y
    return beta_hat


def linear_regression_sklearn(X: NDArray, y: NDArray) -> NDArray:
    """
    Ordinary Least Squares (OLS) linear regression using sklearn.
    """

    model = LinearRegression()
    model.fit(X, y)
    return model.coef_


def ridgeless_regression_sklearn(X: NDArray, y: NDArray) -> NDArray:
    """
    Ridgeless regression using minimum-norm least squares solution.
    Uses pseudoinverse since sklearn doesn't have a Ridgeless class.
    """

    # Use pseudoinverse for minimum-norm solution
    beta_hat = X.T @ np.linalg.inv(X @ X.T) @ y
    return beta_hat


def ridge_regression_sklearn(X: NDArray, y: NDArray, alpha: float) -> NDArray:
    """
    Ridge regression using sklearn.
    """

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model.coef_
