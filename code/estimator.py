from dataclasses import dataclass

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pymanopt
import scipy
import scipy.linalg
import scipy.sparse as sp
import utils
from numpy.linalg import norm, pinv


@dataclass
class Params:
    alpha: float
    threshold_frac: float
    cholesky_noise: float


class Estimator:

    def __init__(self, n: int, params: Params):
        self._alpha = params.alpha
        self._threshold_frac = params.threshold_frac
        self._cholesky_noise = params.cholesky_noise
        self._n = n

        # Init manifold optimization
        self._circle = pymanopt.manifolds.Sphere(2)
        self._manopt_optimizer = pymanopt.optimizers.SteepestDescent()

    def estimate_RE(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        m, n = indices.shape

        data = np.hstack((np.ones(m), -np.ones(m)))
        rows = np.hstack((np.arange(m), np.arange(m)))
        cols = indices.T.reshape(-1)
        inds = (rows, cols)

        C = sp.coo_array((data, inds), shape=(m, self._n), dtype=int).tocsr()

        D_tilde = sp.diags_array(measurements)

        W = sp.diags_array(self._alpha / sigmas)

        D_tilde_sq = D_tilde**2
        Q_1 = D_tilde_sq @ W
        G_pinv = pinv((C.T @ W @ C).toarray())
        Q_2 = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W
        Λ = cp.Variable((m, m), diag=True)

        obj_dual_sdp = cp.Maximize(cp.trace(Λ))

        cons_dual_sdp = [Q_2 - Λ >> 0]

        prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

        prob_dual_sdp.solve("MOSEK", verbose=True)
        assert prob_dual_sdp.value != -np.inf
        Λ_star = sp.dia_array(Λ.value)

        p_sdp_star = Q_1.trace() + Λ_star.trace()

        # Find basis for Y_r
        R_star = Q_2 - Λ_star
        U, Σ, Vt = np.linalg.svd(R_star)
        threshold = Σ.max() * self._threshold_frac  # % of max eigenvalue
        R_rank = np.where(Σ < threshold, 0, 1).sum()

        r = m - R_rank

        print(f"Found {r = }")

        V = Vt.T[:, m - r :]

        assert r != 0

        Z_bar = cp.Variable((r, r), PSD=True)

        obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

        cons_primal = []

        prob_primal = cp.Problem(obj_primal, cons_primal)
        prob_primal.solve("MOSEK", verbose=True)

        # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
        Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value).T

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        @pymanopt.function.autograd(self._circle)
        def cost_mle(Y):
            return np.trace(Q_2 @ Y @ Y.T)

        prob_mle = pymanopt.Problem(self._circle, cost_mle)

        result_mle = self._manopt_optimizer.run(prob_mle, initial_point=Y_0)
        p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point
        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star
        return X_star

    def estimate_RE_mod(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        m, n = indices.shape

        data = np.hstack((np.ones(m), -np.ones(m)))
        rows = np.hstack((np.arange(m), np.arange(m)))
        cols = indices.T.reshape(-1)
        inds = (rows, cols)

        C = sp.coo_array((data, inds), shape=(m, self._n), dtype=int).tocsr()

        D_tilde = sp.diags_array(measurements)

        W = sp.diags_array(self._alpha / sigmas)

        D_tilde_sq = D_tilde**2
        G_pinv = pinv((C.T @ W @ C).toarray())
        Q_2 = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W

        Z = cp.Variable((m, m), PSD=True)

        obj_primal = cp.Minimize(cp.trace(Q_2 @ Z))

        cons_primal = [cp.diag(Z) == 1]

        prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
        prob_primal.solve("MOSEK", verbose=True)

        Y_r_star = scipy.linalg.cholesky(Z.value + np.eye(m) * self._cholesky_noise, lower=True)

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        @pymanopt.function.autograd(self._circle)
        def cost_mle(Y):
            return np.trace(Q_2 @ Y @ Y.T)

        prob_mle = pymanopt.Problem(self._circle, cost_mle)

        result_mle = self._manopt_optimizer.run(prob_mle, initial_point=Y_0)
        p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point
        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star
        return X_star

    def estimate_kruskal(
        self,
        indices: np.ndarray,
        measurements: np.ndarray,
        sigmas: np.ndarray,
        init_guess: np.ndarray,
        max_its: int = 100,
        eps: float = 0.1,
    ):
        m = indices.shape[0]

        def get_grads(X, d_hat, indices):
            """d_hat is measurements"""
            d = utils.get_distances(X, indices)
            diff = d - d_hat

            S_star = np.linalg.norm(diff)
            T_star = np.linalg.norm(d)
            S = np.sqrt(S_star / T_star)

            grads = np.zeros_like(X)

            for k in range(m):
                i, j = indices[k, :]

                # g = S * ((d[k] - d_hat[k]) / S_star - d[k] / T_star) * (X[i, :] - X[j, :]) / d[k]
                g = S * (d[k] - d_hat[k]) * (X[i, :] - X[j, :]) / (d[k] + 1)
                grads[i, :] += g
                grads[j, :] -= g

            return grads

        beta = lambda it: 0.1 / np.sqrt(1 + it)
        d_hat = measurements
        X = init_guess.copy()
        # history = [X.copy()]

        for it in range(max_its):
            grads = get_grads(X, d_hat, indices)
            step = -grads * beta(it)
            X = X + step

            if norm(grads) ** 2 < eps:
                print("Kruskal:", it, "iterations")
                break

        return X


def _main():
    # n = 4
    # m = 6

    # alpha = 0.01
    # sigma = 10
    # sigmas = np.ones(m) * 0.1
    # threshold = 0.01
    # cholesky_noise = 0.001
    # params = Params(alpha, threshold, cholesky_noise)

    # X = 100 * np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)

    # inds = np.array([[0, 1], [1, 0], [0, 2], [0, 3], [2, 3], [3, 1]])
    # sig = sigma * np.ones(m)
    # _d = utils.get_distances(X, inds)
    # meas = utils.generate_measurements(X, inds, sig)

    # estimator = Estimator(n, params)
    # estimator.estimate_kruskal(inds, meas, sig, X @ np.array([[3, 4], [2, -3 / 2]]) / 5, 10)

    # return
    np.random.seed(105)
    np.set_printoptions(precision=3, suppress=True)
    n = 4
    m = n + 8
    assert n * (n - 1) >= m, "Too many measurements!"
    alpha = 0.01
    sigma = 1
    sigmas = np.ones(m) * sigma
    threshold = 0.01
    cholesky_noise = 0.001

    X = utils.generate_positions(n)
    # X = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
    indices = utils.generate_indices(m, n)
    meas = utils.generate_measurements(X, indices, sigmas)

    params = Params(alpha, threshold, cholesky_noise)
    estimator = Estimator(n, params)

    X_hat = estimator.estimate_RE_mod(indices, meas, sigmas)

    X_hat_refined = estimator.estimate_kruskal(
        indices,
        meas,
        sigmas,
        X + np.random.multivariate_normal(np.zeros(2), 10 * np.eye(2), size=n),
        1000,
    )

    utils.plot_unbiased(
        [X, X_hat, X_hat_refined],
        ["Real", "Estimate", "Refined"],
        [False, True, False],
        show=True,
    )


if __name__ == "__main__":
    _main()
