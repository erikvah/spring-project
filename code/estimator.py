from dataclasses import dataclass

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pymanopt
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse as sp
import utils
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman.kalman_filter import KalmanFilter
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

        self._kf = KalmanFilter(4 * n, 2 * n)  # [x1 dx1 x2 dx2 ...]
        # self._kf.Q = scipy.sparse.block_diag(
        #     2 * n * [Q_discrete_white_noise(dim=2, dt=0.5, var=1, order_by_dim=False, block_size=2)]
        # ).tocsr()
        # self._kf.R = scipy.sparse.block_diag(n * [np.eye()])

        pass

    def estimate_RE(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        m, n = indices.shape

        data = np.hstack((-np.ones(m), np.ones(m)))
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

        obj_dual_sdp = cp.Maximize(Λ.trace())

        cons_dual_sdp = [Q_2 - Λ >> 0]

        prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

        solver_params = {
            "mosek_params": {
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1.0,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
            }
        }
        prob_dual_sdp.solve("MOSEK", verbose=True, **solver_params)
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

        assert r >= 2

        Z_bar = cp.Variable((r, r), PSD=True)

        obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

        cons_primal = []

        prob_primal = cp.Problem(obj_primal, cons_primal)
        prob_primal.solve("MOSEK", verbose=True)

        # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
        Y_r_star = (
            V @ scipy.linalg.cholesky(Z_bar.value + self._cholesky_noise * np.eye(r), lower=True).T
        )

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        # Manifold optimization
        manopt_optimizer = pymanopt.optimizers.ConjugateGradient()
        manifold = pymanopt.manifolds.Oblique(m=2, n=m)

        @pymanopt.function.numpy(manifold)
        def cost(Yt):
            # Y is transposed here (2 x m)
            return np.trace(Q_2 @ Yt.T @ Yt)

        @pymanopt.function.numpy(manifold)
        def grad(Yt):
            return Yt @ Q_2.T + Yt @ Q_2

        prob_mle = pymanopt.Problem(manifold, cost=cost, euclidean_gradient=grad)
        # prob_mle = pymanopt.Problem(manifold, cost=cost)

        result_mle = manopt_optimizer.run(prob_mle, initial_point=Y_0.T)
        # p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point.T
        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star

        p_mle_star = utils.get_cost(X_star, indices, measurements, sigmas, self._alpha)

        return X_star, p_mle_star, p_sdp_star

    def estimate_RE_mod(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        raise NotImplementedError()
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

        # Manifold optimization
        manopt_optimizer = pymanopt.optimizers.ConjugateGradient()
        manifold = pymanopt.manifolds.Oblique(m=2, n=m)

        @pymanopt.function.numpy(manifold)
        def cost(Yt):
            # Y is transposed here (2 x m)
            return np.trace(Q_2 @ Yt.T @ Yt)

        @pymanopt.function.numpy(manifold)
        def grad(Yt):
            return Yt @ Q_2.T + Yt @ Q_2

        prob_mle = pymanopt.Problem(manifold, cost=cost, euclidean_gradient=grad)
        # prob_mle = pymanopt.Problem(manifold, cost=cost)

        result_mle = manopt_optimizer.run(prob_mle, initial_point=Y_0.T)
        p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point.T
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
        initial_step: float = 0.1,
        verbose=False,
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
                grads[i, :] += g / sigmas[k]
                grads[j, :] -= g / sigmas[k]

            return grads

        beta = lambda it: initial_step / np.sqrt(1 + it)
        d_hat = measurements
        X = init_guess.copy()

        for it in range(max_its):
            if verbose and (it + 1) % 50 == 0:
                print(f"Kruskal: Iteration {it+1}")
            grads = get_grads(X, d_hat, indices)
            step = -grads * beta(it)
            X = X + step

            if norm(grads) ** 2 < eps:
                print("Kruskal:", it, "iterations")
                break

        return X


def _main():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1213)
    n = 12
    m = 90

    alpha = 0.0001
    sigma = 10
    sigmas = np.ones(m) * sigma
    threshold = 0.03
    cholesky_noise = 0.001
    params = Params(alpha, threshold, cholesky_noise)

    # X = 100 * np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    # indices = np.array([[0, 1], [0, 2], [1, 2], [2, 1]])

    # X = utils.generate_random_positions(n)
    X = utils.generate_grid(4, 3) + np.random.multivariate_normal(
        np.zeros(2), 500 * np.array([[16, 0], [0, 9]], dtype=float), size=n
    )

    N = 12

    utils.plot_unbiased(
        [
            X
            @ np.array(
                [
                    [np.cos(i * np.pi / 6), -np.sin(i * np.pi / 6)],
                    [np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)],
                ]
            ).T
            for i in range(N)
        ],
        list(range(N)),
        show=True,
    )

    return

    indices = utils.generate_indices(m, n)

    d_hat = utils.get_distances(X, indices)
    measurements = utils.generate_measurements(X, indices, sigmas)

    estimator = Estimator(n, params)
    X_hat, cost, l_bound = estimator.estimate_RE(indices, measurements, sigmas)
    # X_tilde = estimator.estimate_RE_mod(indices, measurements, sigmas)

    theta = 0
    for it in range(20):
        X_hat = estimator.estimate_kruskal(
            indices, measurements, sigmas, X_hat, 1000, initial_step=0.3
        )
        utils.plot_unbiased(
            [X, X_hat],
            ["Real", "Estimate"],
            show=True,
        )

        theta += 0.1

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        X = X @ R.T

    return
    # np.random.seed(105)
    # n = 3
    # m = n
    # assert n * (n - 1) >= m, "Too many measurements!"
    # alpha = 0.01
    # sigma = 1
    # sigmas = np.ones(m) * sigma
    # threshold = 0.01
    # cholesky_noise = 0.001

    # X = utils.generate_positions(n)
    # # X = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
    # indices = utils.generate_indices(m, n)
    # meas = utils.generate_measurements(X, indices, sigmas)

    # params = Params(alpha, threshold, cholesky_noise)
    # estimator = Estimator(n, params)

    # X_hat = estimator.estimate_RE_mod(indices, meas, sigmas)

    # X_hat_refined = estimator.estimate_kruskal(
    #     indices,
    #     meas,
    #     sigmas,
    #     X + np.random.multivariate_normal(np.zeros(2), 10 * np.eye(2), size=n),
    #     1000,
    # )

    # utils.plot_unbiased(
    #     [X, X_hat, X_hat_refined],
    #     ["Real", "Estimate", "Refined"],
    #     [False, True, False],
    #     show=True,
    # )


if __name__ == "__main__":
    _main()
