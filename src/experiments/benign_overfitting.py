"""
Benign Overfitting Experiment

Demonstrates the benign overfitting phenomenon in high-dimensional regression
where p >> n (more features than samples). Compares ordinary least squares
(OLS) linear regression with ridgeless regression.

Key Finding:
- Linear regression fails catastrophically when n < p due to singular matrix
- Ridgeless regression produces stable, generalizing solutions despite perfect
  interpolation of noisy training data
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_data
from metrics import calculate_mse, calculate_r2
from models import linear_regression, ridgeless_regression


def main():
    """Run benign overfitting experiment comparing linear vs ridgeless regression."""
    # Generate training data: 50 samples, 200 features (high-dimensional)
    X_train, y_train, beta_true = generate_data(p=200, t=50)

    # Generate test data with same true beta for valid evaluation
    X_test, y_test, _ = generate_data(p=200, t=1000, seed=456, beta=beta_true)

    # Linear regression (OLS) - numerically unstable when t < p
    print("=" * 60)
    print("Linear Regression (OLS)")
    print("=" * 60)
    beta_linear_hat = linear_regression(X_train, y_train)
    prediction_linear = X_test @ beta_linear_hat
    mse = calculate_mse(y_test, prediction_linear)
    r2 = calculate_r2(y_test, prediction_linear)
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print()

    # Ridgeless regression - stable minimum-norm solution
    print("=" * 60)
    print("Ridgeless Regression (Minimum Norm)")
    print("=" * 60)
    beta_ridgeless_hat = ridgeless_regression(X_train, y_train)
    prediction_ridgeless = X_test @ beta_ridgeless_hat
    mse = calculate_mse(y_test, prediction_ridgeless)
    r2 = calculate_r2(y_test, prediction_ridgeless)
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training samples (n): {X_train.shape[0]}")
    print(f"Features (p):         {X_train.shape[1]}")
    print(f"Test samples:         {X_test.shape[0]}")


if __name__ == "__main__":
    main()
