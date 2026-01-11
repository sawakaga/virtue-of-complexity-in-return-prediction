"""
Implementation of Random Fourier Features (RFF) for transforming data into a higher dimensional
feature space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from data import generate_data
from experiments import benign_overfitting
from metrics import calculate_mse, calculate_r2
from models import regression


def get_rff_features(X, output_dim=1000, gamma=0.5, seed=123):
    """
    Transforms X (T x P) into a higher dimensional feature space (T x output_dim).

    1. Generate a random weight matrix 'W' of shape (P, output_dim).
       - Sample from Normal(0, 1) * sqrt(2 * gamma)

    2. Compute Projection = X @ W

    3. Compute Features:
       - We can use [cos(projection), sin(projection)]
       - Or simpler version used in some papers: sqrt(2) * cos(projection + random_bias)

    Let's use the Sin/Cos concatenation method (standard RFF):
    - Z = Concatenate([cos(X @ W), sin(X @ W)])
    - Scale by 1 / sqrt(output_dim)
    """

    generator = np.random.default_rng(seed)
    T, P = X.shape

    # T x output_dim -> 50 x 1000
    w = generator.normal(0, np.sqrt(2 * gamma), size=(P, output_dim))

    # X = T x P -> 50 x 200
    # w = 200 x 1000
    # X @ w -> 50 x 1000
    projection = X @ w

    # 50 x 1000 '+' 50 x 1000 -> 50 x 2000
    Z = np.concatenate([np.cos(projection), np.sin(projection)], axis=1)
    # Normalizing the features to keep the variance of each feature constant
    Z /= np.sqrt(output_dim)

    return Z


def get_rff_features_sklearn(X, output_dim=1000, gamma=0.5, seed=123):
    rbf = RBFSampler(gamma=gamma, n_components=output_dim, random_state=seed)
    return rbf.fit_transform(X)


def main():
    # First run the benign overfitting experiment
    benign_overfitting.main()

    print("\n" + "=" * 60)
    print("RFF EXPERIMENTS")
    print("=" * 60 + "\n")

    # Generate data
    X_train, y_train, beta_true = generate_data()
    X_test, y_test, _ = generate_data(beta=beta_true)

    # Get RFF features using sklearn's implementation
    Z_train_sklearn = get_rff_features_sklearn(X_train)
    Z_test_sklearn = get_rff_features_sklearn(X_test)

    print("=" * 60)
    print("Linear Regression (OLS) with RFF")
    print("=" * 60)
    beta_hat_linear = regression.linear_regression_sklearn(Z_train_sklearn, y_train)
    prediction_linear = Z_test_sklearn @ beta_hat_linear
    linear_mse = calculate_mse(y_test, prediction_linear)
    linear_r2 = calculate_r2(y_test, prediction_linear)

    print(f"Test MSE: {linear_mse:.4f}")
    print(f"Test R²:  {linear_r2:.4f}")
    print()

    print("=" * 60)
    print("Ridgeless Regression with RFF")
    print("=" * 60)
    beta_hat_ridgeless = regression.ridgeless_regression_sklearn(Z_train_sklearn, y_train)
    prediction_ridgeless = Z_test_sklearn @ beta_hat_ridgeless
    ridgeless_mse = calculate_mse(y_test, prediction_ridgeless)
    ridgeless_r2 = calculate_r2(y_test, prediction_ridgeless)

    print(f"Test MSE: {ridgeless_mse:.4f}")
    print(f"Test R²:  {ridgeless_r2:.4f}")


if __name__ == "__main__":
    main()
