from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pymanopt
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse as sp
import utils

# from filterpy.common import Q_discrete_white_noise
# from filterpy.kalman.UKF import UnscentedKalmanFilter
from kalman_filter import EKF
from numpy.linalg import norm, pinv


@dataclass
class EstimatorParams:
    alpha: float  # Scaling in RE
    threshold_frac: float  # Singular values cutoff threshold
    cholesky_noise: float  # Noise to enable stable Cholesky fac
    kalman_Q_gain: float
    kalman_R_gain: float


@dataclass
class EKF_funcs:
    f: Callable
    g: Callable
    A_t: Callable
    C_t: Callable


def get_default_params():
    return EstimatorParams(1, 0.1, 0.01, 1, 1)


def get_default_ekf_funcs(n, dt):
    def separate_xs(x):
        px = x[::3]
        py = x[1::3]
        thetas = x[2::3]

        return px, py, thetas

    def separate_us(u):
        vs = u[::2]
        omegas = u[1::2]

        return vs, omegas

    def A_t(x, u):
        px, py, thetas = separate_xs(x)
        vs, us = separate_us(u)

        As = []
        for i in range(n):
            A = np.eye(3)
            A[0, 2] = -dt * vs[i] * np.sin(thetas[i])
            A[1, 2] = dt * vs[i] * np.cos(thetas[i])
            As.append(A)
        A = scipy.linalg.block_diag(As)

        return A

    def C_t(x):
        Cs = []

        for i in range(n):
            C = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            )

            Cs.append(C)
        C = scipy.linalg.block_diag(Cs)

        return C

    def f(x, u):
        px, py, thetas = separate_xs(x)
        vs, omegas = separate_us(u)

        sint = np.sin(thetas)
        cost = np.cos(thetas)

        x_new = np.zeros_like(x)

        x_new[::3] += dt * vs * cost
        x_new[1::3] += dt * vs * sint
        x_new[2::3] += dt * omegas

    def g(x):
        z = np.zeros(2 * n)

        z[::2] = x[::3]
        z[1::2] = x[1::3]
        return x

    return EKF_funcs(f, g, A_t, C_t)


class Estimator:

    def __init__(self, n: int, params: EstimatorParams, funcs: EKF_funcs):
        self._alpha = params.alpha
        self._threshold_frac = params.threshold_frac
        self._cholesky_noise = params.cholesky_noise
        self._n = n

        # self._ukf = UnscentedKalmanFilter(3 * n, 2 * n,dt, self._g, self._f, )  # [x1 dx1 x2 dx2 ...]
        mu0 = np.zeros(3 * n)
        Sigma0 = np.zeros((3 * n, 3 * n))

        Q = params.kalman_Q_gain * sp.block_diag(
            n * [np.diag([100, 100, 1])]
        )  # 100x more variation in position than angle
        # self._ukf.Q = Q.toarray()

        R = params.kalman_R_gain * sp.block_diag(n * [np.diag([100, 100])])
        # self._ukf.Q = R.toarray()

        self._kf = EKF(funcs.f, funcs.g, funcs.A_t, funcs.C_t, Q, R, mu0, Sigma0)

        self._priors_set = False
        self._do_predict = True

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

        if r >= 2:
            V = Vt.T[:, m - r :]
            Z_bar = cp.Variable((r, r), PSD=True)

            obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

            cons_primal = []

            prob_primal = cp.Problem(obj_primal, cons_primal)
            prob_primal.solve("MOSEK", verbose=True)

            # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
            Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value + self._cholesky_noise * np.eye(r), lower=True).T
        else:
            # Z = cp.Variable((m, m), PSD=True)

            # obj_primal = cp.Minimize(cp.trace(Q_2 @ Z))

            # cons_primal = [cp.diag(Z) == 1]

            # prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
            # prob_primal.solve("MOSEK", verbose=True)

            # Y_r_star = scipy.linalg.cholesky(Z.value + self._cholesky_noise * np.eye(m))
            Z = prob_dual_sdp.solution.dual_vars.popitem()[1]  # type:ignore
            Y_r_star = scipy.linalg.cholesky(Z + self._cholesky_noise * np.eye(m), lower=True)

        Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

        # Manifold optimization
        manifold = pymanopt.manifolds.Oblique(m=2, n=m)

        @pymanopt.function.numpy(manifold)
        def cost(Yt):
            # Y is transposed here (2 x m)
            return np.trace(Q_2 @ Yt.T @ Yt)

        @pymanopt.function.numpy(manifold)
        def grad(Yt):
            return Yt @ Q_2.T + Yt @ Q_2

        @pymanopt.function.numpy(manifold)
        def hess(Yt, H):
            return H @ (Q_2.T + Q_2)

        # manopt_optimizer_rtr = pymanopt.optimizers.TrustRegions()
        # prob_mle_rtr = pymanopt.Problem(manifold, cost=cost, euclidean_gradient=grad, euclidean_hessian=hess)
        # result_mle_rtr = manopt_optimizer_rtr.run(prob_mle_rtr, initial_point=Y_0.T)
        # Y_mle_star = result_mle_rtr.point.T

        manopt_optimizer = pymanopt.optimizers.ConjugateGradient()
        prob_mle = pymanopt.Problem(manifold, cost=cost, euclidean_gradient=grad)
        result_mle = manopt_optimizer.run(prob_mle, initial_point=Y_0.T)
        Y_mle_star = result_mle.point.T

        X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star

        p_mle_star = utils.get_cost(X_star, indices, measurements, sigmas, self._alpha)

        return X_star, p_mle_star, p_sdp_star

    def estimate_gradient(
        self,
        indices: np.ndarray,
        measurements: np.ndarray,
        sigmas: np.ndarray,
        init_guess: np.ndarray,
        max_its: int = 100,
        eps: float = 1e-3,
        init_step: float = 0.1,
        verbose=False,
        kruskal=True,
        l_bound=0,
    ):
        l_bound = max(0, l_bound)
        m = sigmas.shape[0]
        d_hat = measurements
        X = init_guess.copy()

        costs = np.zeros(max_its)

        step_size = init_step
        grads = np.zeros_like(X)
        grads_prev = np.zeros_like(X)

        its_taken = max_its

        for it in range(max_its):
            if verbose and (it + 1) % 50 == 0:
                print(f"Kruskal: Iteration {it+1}")

            costs[it] = utils.get_cost(X, indices, d_hat, sigmas, self._alpha)
            grads[:, :] = self._get_grads(X, d_hat, indices, sigmas)

            if kruskal:
                if it > 0:
                    step_size = self._get_step_kruskal(
                        step_size,
                        grads,
                        grads_prev,
                        costs[it],
                        costs[max(0, it - 1)],
                        costs[max(0, it - 5)],
                    )
            else:
                step_size = 1 / np.sqrt(1 + it)

            grads_prev[:, :] = grads

            step = -grads * step_size
            X = X + step

            if norm(grads) < eps * m:
                # if norm(grads) ** 2 < eps:
                print("Finished after", it, "iterations; gradient is zero")
                its_taken = it
                break

            if costs[it] - l_bound < 10 * eps * m:
                # if norm(grads) ** 2 < eps:
                print("Finished after", it, "iterations; lower bound achieved")
                its_taken = it
                break

        return X, costs[:its_taken]

    def set_priors(self, xs):
        self._kf.μ[::3] = xs[::2]
        self._kf.μ[1::3] = xs[1::2]

        self._kf.Σ[0::3, 0::3] = 100
        self._kf.Σ[1::3, 1::3] = 100
        self._kf.Σ[2::3, 2::3] = np.pi

        self._priors_set = True

    def ekf_predict(self, u):
        assert self._priors_set, "You must set priors before predicting"
        assert self._do_predict, "You cannot predict twice in a row"
        self._do_predict = False

        x, S = self._kf.predict(u)

        return x

    def ekf_update(self, y):
        assert not self._do_predict, "You cannot predict twice in a row"
        self._do_predict = True

        x, S = self._kf.update(y)

        return x

    def _get_step_kruskal(self, step_prev, g, g_prev, cost, cost_prev, cost_5prev):
        # Angle factor
        g_vec = g.reshape(-1)
        g_prev_vec = g_prev.reshape(-1)

        cos_theta = g_vec @ g_prev_vec / (norm(g_vec) * norm(g_prev_vec))

        step_angle = 4.0 ** (cos_theta**3)

        # Relaxation factor
        ratio_5_step = min(1, cost / cost_5prev)

        step_relax = 1.3 / (1 + ratio_5_step**5)

        # Good luck factor
        step_good_luck = min(1, cost / cost_prev)

        return step_prev * step_angle * step_relax * step_good_luck

    def _get_grads(self, X, d_hat, indices, sigmas):
        # d_hat is measurements
        d = utils.get_distances(X, indices)
        diff = d - d_hat

        S_star = np.linalg.norm(diff)
        T_star = np.linalg.norm(d)
        S = np.sqrt(S_star / T_star)

        g = (S * (d - d_hat) * (X[indices[:, 0]] - X[indices[:, 1]]).T / (d + 1)).T
        g = (g.T / sigmas).T

        grads_i = np.zeros_like(X)
        # Needed to prevent buffering (https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
        np.add.at(grads_i, indices[:, 0], g)
        # grads_i_foo[indices[:, 0], :] += g_foo

        grads_j = np.zeros_like(X)
        np.add.at(grads_j, indices[:, 1], -g)
        # grads_j_foo[indices[:, 1], :] -= g_foo

        return grads_i + grads_j
