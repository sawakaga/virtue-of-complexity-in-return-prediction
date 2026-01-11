"""
RMT: Random Matrix Theory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from seaborn import histplot

from data import generate_data


def analyse_eigenvalues(X: NDArray) -> NDArray:
    return np.linalg.eigvalsh(X)

def covariance_matrix(X: NDArray) -> NDArray:
    T,_ = X.shape

    return X.T @ X / T

def plot_eigenvalues(eigenvalues: NDArray):
    plt.figure(figsize = (10,6))
    histplot(
        eigenvalues,
        bins=50,
        stat='density',
        alpha=0.7,
        color='blue',
        label='Eigenvalues',
        kde=False
    )
    plt.title('Distribution of Eigenvalues')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print()

    X_train, _, _ = generate_data()

    cov_matrix = covariance_matrix(X_train)

    eigenvalues = analyse_eigenvalues(cov_matrix)

    plot_eigenvalues(eigenvalues)

if __name__ == "__main__":
    main()
