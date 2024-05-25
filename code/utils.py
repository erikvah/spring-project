import time

import matplotlib.pyplot as plt
import numpy as np

__t0 = 0


def generate_random_positions(n):
    X = np.random.uniform(low=(0, 0), high=(1600, 900), size=(n, 2))

    return X


def generate_grid(w, h):
    ws = 1600 * np.arange(w) / w
    hs = 900 * np.arange(h) / h
    grid_x, grid_y = np.meshgrid(ws, hs)

    X = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    return X


def generate_indices(m, n):
    assert n * (n - 1) >= m, "Too many measurements!"
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


def get_distances(X, indices):
    return generate_measurements(X, indices, np.zeros(indices.shape[0]))


def get_cost(X, indices, measurements, sigmas, alpha):
    distances = get_distances(X, indices)
    return (((alpha / sigmas) / 2) * (distances - measurements) ** 2) @ np.ones_like(sigmas)


def get_unbiased_coords(X):
    mean = X.mean(axis=0)

    _, _, Vt = np.linalg.svd(X - mean)

    return (X - mean) @ Vt.T


def plot_points(Xs: list, labels: list):
    s = 50
    for X, label in zip(Xs, labels):
        plt.scatter(X[:, 0], X[:, 1], label=label, marker="x", s=s)


def plot_unbiased(Xs: list, labels: list, show: bool = True):
    X_unbiased = []
    for X, label in zip(Xs, labels):
        X_unbiased.append(get_unbiased_coords(X))

    plot_points(X_unbiased, labels)
    plt.legend()
    plt.axis("equal")
    if show:
        plt.show()


def tick():
    global __t0
    __t0 = time.time()


def tock():
    t = time.time()
    global __t0
    diff = t - __t0
    return diff
