import matplotlib.pyplot as plt
import numpy as np


def generate_positions(m, n):
    assert n * (n - 1) >= m, "Too many measurements!"
    X = np.random.uniform(low=(0, 0), high=(1600, 900), size=(n, 2))

    return X


def generate_indices(m, n):
    indices = np.zeros((m, 2), dtype=int)

    # Give every node a measurement
    for k in range(n):
        target = np.random.randint(0, n - 1)

        while target == k:
            target = np.random.randint(0, n - 1)

        pair = (k, target)

        indices[k, :] = pair

    for k in range(n, m):
        pair = np.random.choice(n, size=2, replace=False)

        while np.any(np.all(pair == indices[:k], axis=1)):
            pair = np.random.choice(n, size=2, replace=False)

        indices[k, :] = pair

    return indices


def generate_measurements(X, indices, sigmas):
    measurements = np.linalg.norm(X[indices[:, 0]] - X[indices[:, 1]], axis=1) + np.random.normal(
        scale=sigmas
    )

    return measurements


def get_unbiased_coords(X):
    mean = X.mean(axis=0)

    Q, R = np.linalg.qr((X - mean).T)
    return R.T


def plot_points(Xs: list, labels: list):
    for X, label in zip(Xs, labels):
        plt.scatter(X[:, 0], X[:, 1], label=label, marker="x")
