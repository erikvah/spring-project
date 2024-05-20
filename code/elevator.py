import autograd.numpy as anp
import cvxpy as cp
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse as sp
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm, pinv

# Implement sparse matrices!


def main():
    ## Params
    # Graph params
    # n = 5  # Number of vertices
    n = 4  # Number of vertices
    # m = 16  # Number of edges
    m = 5  # Number of edges

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
    # positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]])
    positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])

    assert positions.shape == (n, 2)

    # Incidence matrix
    # C = np.array(
    #     [
    #         [-1, 1, 0, 0, 0],
    #         [-1, 0, 1, 0, 0],
    #         [-1, 0, 0, 0, 1],
    #         [1, -1, 0, 0, 0],
    #         [0, -1, 0, 1, 0],
    #         [0, -1, 0, 0, 1],
    #         [1, 0, -1, 0, 0],
    #         [0, 0, -1, 1, 0],
    #         [0, 0, -1, 0, 1],
    #         [0, 1, 0, -1, 0],
    #         [0, 0, 1, -1, 0],
    #         [0, 0, 0, -1, 1],
    #         [1, 0, 0, 0, -1],
    #         [0, 1, 0, 0, -1],
    #         [0, 0, 1, 0, -1],
    #         [0, 0, 0, 1, -1],
    #     ]
    # )

    C = sp.csr_array(
        [
            [-1, 1, 0, 0],
            [0, -1, 0, 1],
            [0, 0, 1, -1],
            [1, 0, -1, 0],
            [-1, 0, 0, 1],
        ]
    )

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

    cons_dual_sdp = [Q_2 - Λ >> 0]

    prob_dual_sdp = cp.Problem(obj_dual_sdp, cons_dual_sdp)  # type:ignore

    prob_dual_sdp.solve()
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


def get_ij_old(C: np.ndarray, row: int):
    i = np.where(C[row, :] == -1)[0][0]
    j = np.where(C[row, :] == 1)[0][0]

    return i, j


def get_ij(C: sp.csr_array, row: int):
    C_row = C[[row], :].toarray().squeeze()

    i = np.where(C_row == -1)[0][0]
    j = np.where(C_row == 1)[0][0]

    return i, j


if __name__ == "__main__":
    main()
