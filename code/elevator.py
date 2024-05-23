from random import randint

import autograd.numpy as anp
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse as sp
import scipy.special
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm, pinv

np.set_printoptions(precision=1, suppress=True)


def main_old():
    ## Params
    # Graph params
    n = 5  # Number of vertices
    # n = 4  # Number of vertices
    # m = 16  # Number of edges
    m = 11  # Number of edges
    # m = 5  # Number of edges

    # Noise per robot
    σ = np.array(n * [1])
    σ_sq = σ**2

    # Other
    np.random.seed(0)
    α = 100  # Improve matrix conditioning?
    threshold_frac = 0.001
    cholesky_noise = 0.001

    ## Matrices
    # Real positions
    positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]])
    # positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])

    assert positions.shape == (n, 2)

    # Incidence matrix
    C = sp.csr_array(
        [
            [-1, 1, 0, 0, 0],
            # [-1, 0, 1, 0, 0],
            [-1, 0, 0, 0, 1],
            [1, -1, 0, 0, 0],
            # [0, -1, 0, 1, 0],
            [0, -1, 0, 0, 1],
            [1, 0, -1, 0, 0],
            # [0, 0, -1, 1, 0],
            [0, 0, -1, 0, 1],
            [0, 1, 0, -1, 0],
            [0, 0, 1, -1, 0],
            [0, 0, 0, -1, 1],
            # [1, 0, 0, 0, -1],
            # [0, 1, 0, 0, -1],
            [0, 0, 1, 0, -1],
            [0, 0, 0, 1, -1],
        ]
    )

    # C = sp.csr_array(
    #     [
    #         [-1, 1, 0, 0],
    #         [0, -1, 0, 1],
    #         [0, 0, 1, -1],
    #         [1, 0, -1, 0],
    #         [-1, 0, 0, 1],
    #     ]
    # )

    idx_rows, idx_cols = C.nonzero()

    Y_real = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1] / np.sqrt(2)])

    assert np.all(C.sum(axis=1) == 0)  # Sanity checks
    assert C.shape == (m, n)

    # Measurements
    diag = np.zeros(m)

    for it in range(m):
        # i, j = idx_rows[2 * it], idx_cols[2 * it + 1]
        i, j = get_ij(C, it)
        diag[it] = norm(positions[i, :] - positions[j, :]) + np.random.normal(loc=0, scale=σ[i])

    D_tilde = sp.diags_array(diag)

    # Weights
    # W = np.zeros((m, m))

    diag = np.zeros(m)

    for it in range(m):
        # i, j = idx_rows[m], idx_cols[m]
        i, j = get_ij(C, it)
        diag[it] = α / σ_sq[i]

    W = sp.diags_array(diag)

    ## Solve
    D_tilde_sq = D_tilde**2
    Q_1 = D_tilde_sq @ W
    # Q_2 = -D_tilde_sq @ W @ C @ pinv(C.T @ W @ C) @ C.T @ W
    G_pinv = pinv((C.T @ W @ C).toarray())
    Q_2 = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W

    # Dual SDP
    Λ = cp.Variable((m, m), diag=True)

    obj_dual_sdp = cp.Maximize(cp.trace(Λ))

    cons_dual_sdp = [
        Q_2 - Λ >> 0,
        (Q_2 - Λ)[-1, -1] == 0,
    ]

    prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

    prob_dual_sdp.solve("CLARABEL", verbose=True, tol_feas=1e-10)
    assert prob_dual_sdp.value != -np.inf
    Λ_star = sp.dia_array(Λ.value)

    p_sdp_star = Q_1.trace() + Λ_star.trace()

    # Find basis for Y_r
    R_star = Q_2 - Λ_star
    U, Σ, Vt = np.linalg.svd(R_star)
    threshold = Σ.max() * threshold_frac  # % of max eigenvalue
    # Σ_thresh = np.where(Σ < threshold, 0, Σ)
    R_rank = np.where(Σ < threshold, 0, 1).sum()

    # r = m - R_rank
    r = m

    V = Vt.T[:, R_rank:]

    # Solve relaxed primal problem
    if False:
        Z_bar = cp.Variable((r, r), PSD=True)

        obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

        cons_primal = []

        prob_primal = cp.Problem(obj_primal, cons_primal)
        prob_primal.solve()

        # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
        Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value).T

    else:
        Z = cp.Variable((m, m), PSD=True)
        obj_primal = cp.Minimize(cp.trace(Q_2 @ Z))

        cons_primal = [cp.diag(Z) == 1]

        prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
        prob_primal.solve()

        Y_r_star = scipy.linalg.cholesky(Z.value + np.eye(m) * cholesky_noise, lower=True)

    # Solve MLE problem (implement solver?)
    # project to oblique manifold

    Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

    circle = pymanopt.manifolds.Sphere(2)

    @pymanopt.function.autograd(circle)
    def cost_mle(Y):
        return np.trace(Q_2 @ Y @ Y.T)

    prob_mle = pymanopt.Problem(circle, cost_mle)

    optimizer_mle = pymanopt.optimizers.SteepestDescent()
    result_mle = optimizer_mle.run(prob_mle, initial_point=Y_0)
    p_mle_star = result_mle.cost
    Y_mle_star = result_mle.point
    X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star

    # print(p_mle_star - p_sdp_star)
    # print(result_mle.point)
    print(positions)
    print(X_star - X_star[0, :])

    return


def main():
    np.random.seed(0)
    n = 10
    m = n + 60
    assert n * (n - 1) >= m, "Too many measurements!"
    alpha = 10
    sigma = 1
    threshold = 0.001
    cholesky_noise = 0.001

    X = np.random.uniform(low=(0, 0), high=(1600, 900), size=(n, 2))
    # X = np.random.multivariate_normal(
    #     mean=np.zeros(2), cov=10_000 * np.array([[3, 0], [0, 1]]), size=n
    # )

    # X = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])

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

    sigmas = np.ones(m) * sigma

    measurements = np.linalg.norm(X[indices[:, 0]] - X[indices[:, 1]], axis=1) + np.random.normal(
        scale=sigmas
    )

    X_hat = solve_elevator(indices, measurements, sigmas, m, n, alpha, threshold, cholesky_noise)

    X_mean = X.mean(axis=0)
    X_hat_mean = X_hat.mean(axis=0)

    X = X - X_mean
    X_hat = X_hat - X_hat_mean

    # foo, X_var, X_ax = np.linalg.svd(X, full_matrices=False)
    # _, X_hat_var, X_hat_ax = np.linalg.svd(X_hat, full_matrices=False)
    Q1, R1 = np.linalg.qr(X.T)
    if np.linalg.det(Q1) < 0:
        Q1[:, [0, 1]] = Q1[:, [1, 0]]
        R1[[0, 1], :] = R1[[1, 0], :]

    if Q1[0, 0] < 0:
        Q1 = -Q1
        R1 = -R1

    X_normalized = R1.T

    Q2, R2 = np.linalg.qr(X_hat.T)
    if np.linalg.det(Q2) < 0:
        Q2[:, [0, 1]] = Q2[:, [1, 0]]
        R2[[0, 1], :] = R2[[1, 0], :]

    if Q2[0, 0] < 0:
        Q2 = -Q2
        R2 = -R2

    X_hat_normalized = R2.T

    # X_normalized = (X_ax.T @ X.T @ foo.T).T
    # X_hat_normalized = (X_ax @ np.linalg.inv(X_hat_ax) @ (X_hat - X_hat_mean).T).T
    # X_hat_normalized = (np.linalg.inv(X_hat_ax) @ (X_hat - X_hat_mean).T).T

    plt.scatter(X_normalized[:, 0], X_normalized[:, 1], marker="x", label="Real")
    plt.scatter(X_hat_normalized[:, 0], X_hat_normalized[:, 1], marker="+", label="Estimate")

    # plt.scatter((X - X_mean)[:, 0], (X - X_mean)[:, 1], marker="x", label="Real")
    # plt.scatter(
    #     (X_hat - X_hat_mean)[:, 0], (X_hat - X_hat_mean)[:, 1], marker="x", label="Estimate"
    # )
    # plt.scatter(X_hat_normalized[:, 0], X_hat_normalized[:, 1], marker="x", label="Estimate (Corr)")
    # plt.scatter(X_normalized[:, 0], X_normalized[:, 1], marker="x", label="Real")
    # plt.scatter(X_hat_normalized[:, 0], X_hat_normalized[:, 1], label="Estimate")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return


def get_ij_old(C: np.ndarray, row: int):
    i = np.where(C[row, :] == -1)[0][0]
    j = np.where(C[row, :] == 1)[0][0]

    return i, j


def get_ij(C: sp.csr_array, row: int):
    C_row = C[[row], :].toarray().squeeze()

    i = np.where(C_row == -1)[0][0]
    j = np.where(C_row == 1)[0][0]

    return i, j


def solve_elevator(
    indices: np.ndarray,
    measurements: np.ndarray,
    sigmas: np.ndarray,
    m: int,
    n: int,
    alpha: float,
    threshold_frac: float,
    cholesky_noise: float,
) -> np.ndarray:
    data = np.hstack((np.ones(m), -np.ones(m)))
    rows = np.hstack((np.arange(m), np.arange(m)))
    cols = indices.T.reshape(-1)
    inds = (rows, cols)

    C = sp.coo_array((data, inds), shape=(m, n), dtype=int).tocsr()

    D_tilde = sp.diags_array(measurements)

    W = sp.diags_array(alpha / sigmas)

    D_tilde_sq = D_tilde**2
    Q_1 = D_tilde_sq @ W
    # Q_2 = -D_tilde_sq @ W @ C @ pinv(C.T @ W @ C) @ C.T @ W
    G_pinv = pinv((C.T @ W @ C).toarray())
    Q_2 = -D_tilde_sq @ W @ C @ G_pinv @ C.T @ W

    Λ = cp.Variable((m, m), diag=True)

    obj_dual_sdp = cp.Maximize(cp.trace(Λ))

    cons_dual_sdp = [Q_2 - Λ >> 0]

    prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

    prob_dual_sdp.solve("MOSEK")
    assert prob_dual_sdp.value != -np.inf
    Λ_star = sp.dia_array(Λ.value)

    p_sdp_star = Q_1.trace() + Λ_star.trace()

    # Find basis for Y_r
    R_star = Q_2 - Λ_star
    U, Σ, Vt = np.linalg.svd(R_star)
    threshold = Σ.max() * threshold_frac  # % of max eigenvalue
    # Σ_thresh = np.where(Σ < threshold, 0, Σ)
    R_rank = np.where(Σ < threshold, 0, 1).sum()

    r = m - R_rank
    V = Vt.T[:, m - r :]

    if r == 0:
        Z = cp.Variable((m, m), PSD=True)

        obj_primal = cp.Minimize(cp.trace(Q_2 @ Z))

        cons_primal = [cp.diag(Z) == 1]

        prob_primal = cp.Problem(obj_primal, cons_primal)  # type:ignore
        prob_primal.solve()

        Y_r_star = scipy.linalg.cholesky(Z.value + np.eye(m) * cholesky_noise, lower=True)

    else:
        print(f"{r = }")

        Z_bar = cp.Variable((r, r), PSD=True)

        obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

        cons_primal = []

        prob_primal = cp.Problem(obj_primal, cons_primal)
        prob_primal.solve()

        # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
        Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value).T

    Y_0 = (Y_r_star[:, 0:2].T / norm(Y_r_star[:, 0:2], axis=1)).T

    circle = pymanopt.manifolds.Sphere(2)

    @pymanopt.function.autograd(circle)
    def cost_mle(Y):
        return np.trace(Q_2 @ Y @ Y.T)

    prob_mle = pymanopt.Problem(circle, cost_mle)

    optimizer_mle = pymanopt.optimizers.SteepestDescent()
    result_mle = optimizer_mle.run(prob_mle, initial_point=Y_0)
    p_mle_star = result_mle.cost
    Y_mle_star = result_mle.point
    X_star = G_pinv @ C.T @ W @ D_tilde @ Y_mle_star
    return X_star


if __name__ == "__main__":
    main()
