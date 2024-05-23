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

    def __init__(self, m: int, n: int, params: Params):
        self._alpha = params.alpha
        self._threshold_frac = params.threshold_frac
        self._cholesky_noise = params.cholesky_noise
        self._n = n
        self._m = m

        self._Λ = cp.Variable((m, m), diag=True)
        self._Q_2 = cp.Parameter((m, m))

        # Init dual SDP
        obj_dual_sdp = cp.Maximize(cp.trace(self._Λ))
        cons_dual_sdp = [self._Q_2 - self._Λ >> 0]
        self._prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

        # Init manifold optimization
        self._circle = pymanopt.manifolds.Sphere(2)
        self._manopt_optimizer = pymanopt.optimizers.SteepestDescent()

    def estimate_RE(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        data = np.hstack((np.ones(self._m), -np.ones(self._m)))
        rows = np.hstack((np.arange(self._m), np.arange(self._m)))
        cols = indices.T.reshape(-1)
        inds = (rows, cols)

        C = sp.coo_array((data, inds), shape=(self._m, self._n), dtype=int).tocsr()

        D_tilde = sp.diags_array(measurements)

        W = sp.diags_array(self._alpha / sigmas)

        D_tilde_sq = D_tilde**2
        Q_1 = D_tilde_sq @ W
        G_pinv = pinv((C.T @ W @ C).toarray())
        self._Q_2.value = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W

        self._prob_dual_sdp.solve("MOSEK", verbose=False)
        assert self._prob_dual_sdp.value != -np.inf
        Λ_star = sp.dia_array(self._Λ.value)

        p_sdp_star = Q_1.trace() + Λ_star.trace()

        # Find basis for Y_r
        R_star = self._Q_2.value - Λ_star
        U, Σ, Vt = np.linalg.svd(R_star)
        threshold = Σ.max() * self._threshold_frac  # % of max eigenvalue
        # Σ_thresh = np.where(Σ < threshold, 0, Σ)
        R_rank = np.where(Σ < threshold, 0, 1).sum()

        r = self._m - R_rank

        V = Vt.T[:, self._m - r :]

        assert r != 0
        # if r == 0:
        #     Z = cp.Variable((m, m), PSD=True)

        #     obj_primal = cp.Minimize(cp.trace(Q_2 @ Z))

        #     cons_primal = [cp.diag(Z) == 1]

        #     prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
        #     prob_primal.solve()

        #     Y_r_star = scipy.linalg.cholesky(Z.value + np.eye(m) * cholesky_noise, lower=True)

        Z_bar = cp.Variable((r, r), PSD=True)

        obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(self._m))))

        cons_primal = []

        prob_primal = cp.Problem(obj_primal, cons_primal)
        prob_primal.solve()

        # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
        Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value).T

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        @pymanopt.function.autograd(self._circle)
        def cost_mle(Y):
            return np.trace(self._Q_2.value @ Y @ Y.T)

        prob_mle = pymanopt.Problem(self._circle, cost_mle)

        result_mle = self._manopt_optimizer.run(prob_mle, initial_point=Y_0)
        p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point
        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star
        return X_star

    def estimate_RE_mod(self, indices: np.ndarray, measurements: np.ndarray, sigmas: np.ndarray):
        data = np.hstack((np.ones(self._m), -np.ones(self._m)))
        rows = np.hstack((np.arange(self._m), np.arange(self._m)))
        cols = indices.T.reshape(-1)
        inds = (rows, cols)

        C = sp.coo_array((data, inds), shape=(self._m, self._n), dtype=int).tocsr()

        D_tilde = sp.diags_array(measurements)

        W = sp.diags_array(self._alpha / sigmas)

        D_tilde_sq = D_tilde**2
        G_pinv = pinv((C.T @ W @ C).toarray())
        self._Q_2.value = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W

        Z = cp.Variable((m, m), PSD=True)

        obj_primal = cp.Minimize(cp.trace(self._Q_2.value @ Z))

        cons_primal = [cp.diag(Z) == 1]

        prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
        prob_primal.solve()

        Y_r_star = scipy.linalg.cholesky(Z.value + np.eye(m) * cholesky_noise, lower=True)

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        @pymanopt.function.autograd(self._circle)
        def cost_mle(Y):
            return np.trace(self._Q_2.value @ Y @ Y.T)

        prob_mle = pymanopt.Problem(self._circle, cost_mle)

        result_mle = self._manopt_optimizer.run(prob_mle, initial_point=Y_0)
        p_mle_star = result_mle.cost
        Y_mle_star = result_mle.point
        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star
        return X_star


if __name__ == "__main__":
    np.random.seed(0)
    n = 10
    m = n + 20
    assert n * (n - 1) >= m, "Too many measurements!"
    alpha = 0.0001
    sigma = 1
    sigmas = np.ones(m) * sigma
    threshold = 0.01
    cholesky_noise = 0.001

    X = utils.generate_positions(m, n)
    indices = utils.generate_indices(m, n)
    meas = utils.generate_measurements(X, indices, sigmas)

    params = Params(alpha, threshold, cholesky_noise)
    estimator = Estimator(m, n, params)

    X_hat = estimator.estimate_RE_mod(indices, meas, sigmas)

    X_tilde = utils.get_unbiased_coords(X)
    X_tilde_hat = utils.get_unbiased_coords(X_hat)

    utils.plot_points([X_tilde, X_tilde_hat], ["Real", "Estimate"])
    plt.legend()
    plt.show()
