import numpy as np
from numpy.typing import NDArray


def generate_data(t=50, p=200, signal_dims=10, noise_std=0.1, seed=123, beta=None):
    '''
    Generate data for benign overfitting experiment.
    T = Observation
    P = Feature
    X = Normal distribution
    Y_target = BetaX + Normal noise
    '''

    generator = np.random.default_rng(seed)

    if beta is None:
        beta = np.zeros(p)
        beta[:signal_dims] = generator.normal(0, 1, signal_dims)

    X = generator.normal(0, 1, size=(t, p))
    y = X @ beta + generator.normal(0, noise_std, t)
    return X, y, beta

def calculate_mse(y_true: NDArray, y_pred: NDArray):
    return np.mean((y_true - y_pred)**2)

def calculate_r2(y_true: NDArray, y_pred: NDArray):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

def linear_regression(X: NDArray, y: NDArray):
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat

def ridgeless_regression(X: NDArray, y: NDArray):
    beta_hat = X.T @ np.linalg.inv(X @ X.T) @ y
    return beta_hat

if __name__ == "__main__":
    X_train, y_train, beta_true = generate_data(p=200, t=50)
    X_test, y_test, _ = generate_data(p=200, t=1000, seed=456, beta=beta_true)

    beta_linear_hat = linear_regression(X_train, y_train)
    prediction_linear =  X_test @ beta_linear_hat
    mse = calculate_mse(y_test, prediction_linear)
    r2 = calculate_r2(y_test, prediction_linear)
    print(f"MSE for linear regression: {mse:.4f}")
    print(f"R2 for linear regression: {r2:.4f}")

    beta_ridgeless_hat = ridgeless_regression(X_train, y_train)
    prediction_ridgeless = X_test @ beta_ridgeless_hat
    mse = calculate_mse(y_test, prediction_ridgeless)
    r2 = calculate_r2(y_test, prediction_ridgeless)
    print(f"MSE for ridgeless regression: {mse:.4f}")
    print(f"R2 for ridgeless regression: {r2:.4f}")
