"""
Implementation of Random Fourier Features (RFF) for transforming data into a higher dimensional
feature space.
"""

from socket import MSG_EOF
import sys
from pathlib import Path

from pandas.core.groupby import base

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

    X_train, y_train, beta_true = generate_data()
    X_test, y_test, _ = generate_data(beta=beta_true)


    # We use the Ridgeless Linear Regression as the bar to beat
    beta_ridgeless = regression.ridgeless_regression_sklearn(X_train, y_train)
    pred_base = X_test @ beta_ridgeless
    base_r2 = calculate_r2(y_test, pred_base)
    base_mse = calculate_mse(y_test, pred_base)

    print(f"BASELINE (Linear Ridgeless) R2: {base_r2:.4f}")
    print(f"BASELINE (Linear Ridgeless) MSE: {base_mse:.4f}")
    print("-" * 50)
    print(f"{'Gamma':<10} | {'Test R2':<10} | {'Status R2':<10} | {'Test MSE':<10} | {'Status MSE':<10}")
    print("-" * 50)


    # We try different "zoom levels" - gamma
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    for g in gammas:
        Z_train = get_rff_features_sklearn(X_train, output_dim=1000, gamma=g, seed=123)
        Z_test  = get_rff_features_sklearn(X_test,  output_dim=1000, gamma=g, seed=123)

        # B. Solve (Ridgeless)
        beta_rff = regression.ridgeless_regression_sklearn(Z_train, y_train)

        # C. Predict
        pred_rff = Z_test @ beta_rff
        r2 = calculate_r2(y_test, pred_rff)
        mse = calculate_mse(y_test, pred_rff)

        # Check if we beat the baseline
        status_r2 = "WINNER" if r2 > base_r2 else "LOSE"
        status_mse = "WINNER" if mse < base_mse else "LOSE"

        print(f"{g:<10} | {r2:.4f}     | {status_r2} | {mse:.4f}     | {status_mse}")

if __name__ == "__main__":
    main()
