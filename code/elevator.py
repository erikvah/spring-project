import autograd.numpy as anp
import cvxpy as cp
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import scipy
import scipy.linalg
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
    α = 10  # Improve matrix conditioning?
    threshold_frac = 0.001

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

    C = np.array(
        [
            [-1, 1, 0, 0],
            [0, -1, 0, 1],
            [0, 0, 1, -1],
            [1, 0, -1, 0],
            [-1, 0, 0, 1],
        ]
    )
    assert np.all(np.sum(C, axis=1) == 0)  # Sanity checks
    assert C.shape == (m, n)

    # Measurements
    diag = np.zeros(m)

    for it in range(m):
        i, j = get_ij(C, it)
        diag[it] = norm(positions[i, :] - positions[j, :]) + np.random.normal(loc=0, scale=σ[i])

    D_tilde = sp.diags_array(diag)

    # Weights
    # W = np.zeros((m, m))

    diag = np.zeros(m)

    for it in range(m):
        i, j = get_ij(C, it)
        diag[it] = α / σ_sq[i]

    W = sp.diags_array(diag)

    ## Solve
    D_tilde_sq = D_tilde**2
    Q_1 = D_tilde_sq @ W
    G = W @ C @ pinv(C.T @ W @ C) @ C.T @ W
    Q_2 = -D_tilde_sq @ G

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
    r = m - R_rank

    V = Vt.T[:, R_rank:]

    # Solve relaxed primal problem
    Z_bar = cp.Variable((r, r), PSD=True)

    obj_primal = cp.Minimize(cp.square(cp.norm2(cp.diag(V @ Z_bar @ V.T) - np.ones(m))))

    cons_primal = []

    prob_primal = cp.Problem(obj_primal, cons_primal)
    prob_primal.solve()

    # Y_r_star = scipy.linalg.cholesky(Z_bar.value) @ V  # Error in paper?
    Y_r_star = V @ scipy.linalg.cholesky(Z_bar.value).T

    # Solve MLE problem (implement solver?)
    # project to oblique manifold

    Y_0 = (Y_r_star.T / norm(Y_r_star, axis=1)).T

    circle = pymanopt.manifolds.Sphere(2)

    @pymanopt.function.autograd(circle)
    def cost_mle(Y):
        return np.trace(Q_2 @ Y @ Y.T)

    prob_mle = pymanopt.Problem(circle, cost_mle)

    optimizer_mle = pymanopt.optimizers.SteepestDescent()
    result_mle = optimizer_mle.run(prob_mle, initial_point=Y_0)
    p_mle_star = result_mle.cost
    Y_mle_star = result_mle.point

    print(p_mle_star - p_sdp_star)
    print(result_mle.point)

    return


def get_ij(C: np.ndarray, row: int):
    i = np.where(C[row, :] == -1)[0][0]
    j = np.where(C[row, :] == 1)[0][0]

    return i, j


if __name__ == "__main__":
    main()
