import numpy as np


def generate_data(
    t=50, p=200, signal_dims=10, noise_std=0.1, seed=123, beta=None, complexity_mode=False
):
    """
    Generate synthetic data for benign overfitting experiments.

    Parameters
    ----------
    t : int, default=50
        Number of observations (samples)
    p : int, default=200
        Number of features
    signal_dims : int, default=10
        Number of features with true signal (non-zero coefficients)
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise added to target variable
    seed : int, default=123
        Random seed for reproducibility
    beta : NDArray, optional
        True coefficients. If None, generates random sparse beta.
        Shape should be (p,)
    """
    generator = np.random.default_rng(seed)

    if beta is None:
        beta = np.zeros(p)
        beta[:signal_dims] = generator.normal(0, 1, signal_dims)

    X = generator.normal(0, 1, size=(t, p))
    linear_signal = X @ beta

    if complexity_mode:
        # The relationship is now a Parabola, not a Line.
        signal = (linear_signal / np.sqrt(signal_dims)) ** 2
        y = signal + generator.normal(0, noise_std, t)
    else:
        # Old Linear Mode
        y = linear_signal + generator.normal(0, noise_std, t)

    return X, y, beta
