import numpy as np


def gaussian(mu: float, sigma: float):
    """Returns a gaussian noise generator"""
    return lambda: np.random.normal(loc=mu, scale=sigma)


if __name__ == "__main__":
    noise = gaussian(0, 1)

    print(noise())
    print(noise())
    print(noise())
