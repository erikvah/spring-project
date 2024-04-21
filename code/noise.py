import numpy as np


def gaussian(mu: np.ndarray, Sigma: np.ndarray):
    """Returns a gaussian noise generator"""
    return lambda: np.random.multivariate_normal(mean=mu, cov=Sigma)


if __name__ == "__main__":
    noise = gaussian(np.array([0, 0]), np.eye(2))

    print(noise())
    print(noise())
    print(noise())
