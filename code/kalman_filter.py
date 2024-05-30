import numpy as np
from scipy.linalg import sqrtm


class KalmanFilter:
    def __init__(self, A, B, C, Q, R, μ0, Σ0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.μ = μ0
        self.Σ = Σ0

        self._next_predict = True

    def predict(self, u):
        assert self._next_predict
        self._next_predict = False

        self.μ = self.A @ self.μ + self.B @ u
        self.Σ = self.A @ self.Σ @ self.A.T + self.Q

        return self.μ, self.Σ

    def update(self, y):
        assert not self._next_predict
        self._next_predict = True

        self.K = self.Σ @ self.C.T @ np.linalg.inv(self.C @ self.Σ @ self.C.T + self.R)
        self.μ = self.μ + self.K @ (y - self.C @ self.μ)
        self.Σ = self.Σ - self.K @ self.C @ self.Σ

        return self.μ, self.Σ


class EKF:
    def __init__(self, f, g, A_func, C_func, Q, R, μ0, Σ0):
        self.f = f
        self.g = g
        self.A = A_func
        self.C = C_func
        self.Q = Q
        self.R = R
        self.μ = μ0
        self.Σ = Σ0

        self._next_predict = True

    def predict(self, u):
        assert self._next_predict
        self._next_predict = False

        self.μ = self.f(self.μ, u)
        self.Σ = self.A(self.μ, u) @ self.Σ @ self.A(self.μ, u).T + self.Q

        return self.μ, self.Σ

    def update(self, y):
        assert not self._next_predict
        self._next_predict = True

        # self.K = self.Σ @ self.C(self.μ).T @ np.linalg.inv(self.C(self.μ) @ self.Σ @ self.C(self.μ).T + self.R)
        self.K = np.linalg.solve(self.C(self.μ) @ self.Σ @ self.C(self.μ).T + self.R, self.C(self.μ) @ self.Σ).T
        self.μ = self.μ + self.K @ (y - self.g(self.μ))
        self.Σ = self.Σ - self.K @ self.C(self.μ) @ self.Σ

        self.μ = np.round(self.μ, 5)
        self.Σ = np.round(self.Σ, 5)
        self.K = np.round(self.K, 5)

        # We expect Σ to be a valid covariance matrix, so we symmetrize
        self.Σ = 0.5 * (self.Σ + self.Σ.T)

        return self.μ, self.Σ


class UKF:
    def __init__(self, f, g, Q, R, μ0, Σ0):
        self.f = f
        self.g = g
        self.Q = Q
        self.R = R
        self.μ = μ0
        self.Σ = Σ0

        self._next_predict = True

    def predict(self, u):
        assert self._next_predict
        self._next_predict = False

        n = self.μ.shape[0]

        xs, ws = self._UT(self.μ, self.Σ)
        xs_bar = np.zeros_like(xs)
        for i in range(2 * n + 1):
            xs_bar[:, i] = self.f(xs[:, i], u)

        self.μ, self.Σ = self._IUT(xs_bar, ws)
        self.Σ += self.Q

        return self.μ, self.Σ

    def update(self, y):
        assert not self._next_predict
        self._next_predict = True

        n = self.μ.shape[0]
        m = self.R.shape[0]

        xs, ws = self._UT(self.μ, self.Σ)
        ys = np.zeros((m, 1 + 2 * n))
        for i in range(2 * n + 1):
            ys[:, i] = self.g(xs[:, i])

        ys_hat, Σ_y = self._IUT(ys, ws)
        Σ_y += self.R

        Σ_xy = (ws * (xs - self.μ[:, np.newaxis])) @ (ys - ys_hat[:, np.newaxis]).T

        self.μ = self.μ + Σ_xy @ np.linalg.solve(Σ_y, y - ys_hat)
        self.Σ = self.Σ - Σ_xy @ np.linalg.solve(Σ_y, Σ_xy.T)

        # We expect Σ to be a valid covariance matrix, so we symmetrize
        self.Σ = 0.5 * (self.Σ + self.Σ.T)

        return self.μ, self.Σ

    @staticmethod
    def _UT(μ, Σ, λ=2):
        n = μ.shape[0]
        x_0 = μ.reshape(-1, 1)
        w_0 = np.array(λ / (λ + n))

        M = sqrtm((λ + n) * Σ)

        x_l = (μ + M.T).T  # x_i on cols
        w_l = np.array(n * [1 / (2 * (λ + n))])

        x_u = (μ - M.T).T  # x_i on cols
        w_u = np.array(n * [1 / (2 * (λ + n))])

        return np.hstack((x_0, x_l, x_u)), np.hstack((w_0, w_l, w_u))

    @staticmethod
    def _IUT(xs, ws):
        μ = (xs * ws).sum(axis=1, keepdims=True)
        Σ = (ws * (xs - μ)) @ (xs - μ).T

        return μ.squeeze(), Σ


class PF:
    def __init__(self, f, g, Q, R, x0):
        """Assuming Gaussian noise"""
        self.f = f
        self.g = g
        self.Q = Q
        self.R = R

        N = x0.shape[0]
        self.N = N

        self.ws = np.ones(N) / N
        self.xs = x0

    def predict(self, u):
        new_xs = np.zeros_like(self.xs)

        for i in range(self.N):
            new_xs[i, :] = self.f(self.xs[i, :], u) + np.random.multivariate_normal(
                np.zeros_like(self.xs[i, :]), self.Q
            )

        self.xs = new_xs

    def update(self, y):
        w_hats = np.zeros_like(self.ws)
        for i in range(self.N):
            xi = self.xs[i, :]
            w_hats[i] = np.exp(-0.5 * (y - self.g(xi)) @ np.linalg.solve(self.R, y - self.g(xi)))

        self.ws = w_hats / w_hats.sum()

        self._importance_resample()

        return self.xs, self.ws

    def _importance_resample(self):
        samples = np.random.rand(self.N)
        bins = np.cumsum(self.ws)
        x_idxs = np.zeros_like(bins, dtype=int)

        for k, sample in enumerate(samples):
            i = 0
            while sample > bins[i]:
                i += 1

            x_idxs[k] = i

        self.xs = self.xs[x_idxs, :]
        self.ws[:] = 1 / self.N


if __name__ == "__main__":
    ukf = UKF(None, None, None, None, None, None)

    mu = np.array((1, 0))
    S = np.diag([5, 5])

    xs, ωs = ukf._UT(mu, S)
    μ, Σ = ukf._IUT(xs, ωs)

    assert np.isclose(μ, mu).all()
    assert np.isclose(Σ, S).all()

    for i in range(10, 100):
        mu = np.random.randn(i)
        F = np.random.randn(i, i)
        S = F.T @ F

        xs, ωs = ukf._UT(mu, S)
        μ, Σ = ukf._IUT(xs, ωs)

        assert np.isclose(μ, mu).all()
        assert np.isclose(Σ, S).all()
