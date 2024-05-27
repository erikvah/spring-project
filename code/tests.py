import matplotlib.pyplot as plt
import numpy as np
import utils
from estimator import Estimator, get_default_params


def run_kruskal_tests():
    np.random.seed(578)
    ns = [4, 10]
    # ns = [50]
    ms = [[5, 11], [40, 85]]
    # ms = [[200, 500]]
    # grids = [(2, 2), (5, 4)]
    grids = [(2, 2), (5, 2)]

    for it, (n, grid) in enumerate(zip(ns, grids)):
        for m in ms[it]:
            trial_kruskal, trial_naive = kruskal_test(m, n, 5, *grid)

            move_indices_n = [0]
            running_sum = 0
            for cost_hist in trial_naive:
                running_sum += len(cost_hist)
                move_indices_n.append(running_sum - 1)

            move_indices_k = [0]
            running_sum = 0

            for cost_hist in trial_kruskal:
                running_sum += len(cost_hist)
                move_indices_k.append(running_sum - 1)

            flat_hist_n = [cost for cost_hist in trial_naive for cost in cost_hist]
            flat_hist_k = [cost for cost_hist in trial_kruskal for cost in cost_hist]

            fig, axs = plt.subplots(2, 1, sharey=True, figsize=(8, 4))

            ax_n, ax_k = axs

            ax_k.step(range(len(flat_hist_k)), flat_hist_k)
            ax_n.step(range(len(flat_hist_n)), flat_hist_n)

            for move_index in move_indices_n:
                ax_n.axvline(move_index, color="C1", linestyle="--")

            for move_index in move_indices_k:
                ax_k.axvline(move_index, color="C1", linestyle="--")

            ax_n.set_title(f"Target tracking with gradient descent (${m = }, {n = }$)")
            ax_n.legend([f"$\\frac{{1}}{{\\sqrt{{k}}}}$ cost", "Robots move"])
            ax_n.set_yscale("log")
            ax_n.set_xlabel("Iterations")
            ax_n.set_ylabel("Stress")
            ax_k.legend(["Kruskal cost", "Robots move"])
            ax_k.set_yscale("log")
            ax_k.set_xlabel("Iterations")
            ax_k.set_ylabel("Stress")
            plt.show()

    # for cost_hist in cost_hists:
    #     plt.plot()


def kruskal_test(m, n, N, w, h):
    params = get_default_params()
    estimator = Estimator(n, params)

    sigmas = utils.generate_sigmas(m)
    X = utils.generate_grid(w, h)
    X_hat_kruskal = X + np.random.multivariate_normal(np.zeros(2), 10 * np.eye(2), size=n)
    X_hat_naive = X + np.random.multivariate_normal(np.zeros(2), 10 * np.eye(2), size=n)
    indices = utils.generate_indices(m, n)
    Y = utils.generate_measurements(X, indices, sigmas)

    cost_hists_kruskal = []
    cost_hists_naive = []

    speed = np.random.multivariate_normal(np.zeros(2), 100 * np.eye(2), size=n)

    for it in range(N):
        # X += np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
        X += speed
        indices = utils.generate_indices(m, n)
        Y = utils.generate_measurements(X, indices, sigmas)

        X_hat_kruskal, cost_hist_kruskal = estimator.estimate_gradient(
            indices, Y, sigmas, X_hat_kruskal, kruskal=True, max_its=200, init_step=0.05
        )
        X_hat_naive, cost_hist_naive = estimator.estimate_gradient(
            indices, Y, sigmas, X_hat_naive, kruskal=False, max_its=200, init_step=0.05
        )

        cost_hists_kruskal.append(cost_hist_kruskal)
        cost_hists_naive.append(cost_hist_naive)

    fig = plt.figure()
    ax = fig.gca()
    utils.plot_points(
        [X, X_hat_kruskal, X_hat_naive],
        ["Real", "Kruskal", "Naive"],
        markers=3 * ["x"],
        sizes=3 * [100],
        colors=["C1", "C2", "C3"],
        ax=ax,
    )
    plt.legend()
    plt.show()

    return cost_hists_kruskal, cost_hists_naive


def run_RE_tests():
    np.random.seed(3141)
    ns = [4, 10]
    ms = [[5, 11], [40, 85]]
    grids = [(2, 2), (5, 2)]

    # ns = [4]
    # ms = [[11]]
    # grids = [(2, 2)]

    for it, (n, grid) in enumerate(zip(ns, grids)):
        for m in ms[it]:
            X, X_hat, X_hat_refined, indices = RE_test(m, n, *grid)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()

            X_unbiased = utils.plot_unbiased(
                ax,
                [X, X_hat, X_hat_refined],
                [None, None, None],
                markers=["x", "+", "+"],
                sizes=[0, 0, 0],
                colors=["C2", "C3", "C1"],
            )

            utils.plot_measurements(X_unbiased, indices, ["Real", "Estimate", "Refined"], ["C2", "C3", "C1"], ax)

            X_unbiased = utils.plot_unbiased(
                ax,
                [X, X_hat, X_hat_refined],
                ["Real", "Estimate", "Refined"],
                markers=["x", "+", "+"],
                sizes=[200, 200, 200],
                colors=["C2", "C3", "C1"],
            )

            ax.set_title(f"Point estimation (${m = }$, ${n = }$)")
            plt.legend()
            plt.show()


def RE_test(m, n, w, h):
    assert n * (n - 1) >= m, "Too many measurements!"

    X_pos_noise = np.random.multivariate_normal(np.zeros(2), 100**2 * np.eye(2), size=n)
    X = utils.generate_grid(w, h) + X_pos_noise
    sigmas = utils.generate_sigmas(m, max=10)
    indices = utils.generate_indices(m, n)
    Y = utils.generate_measurements(X, indices, sigmas)

    # max_r = 1000
    # max_sigma = 50
    # Y, indices, sigmas = utils.generate_realistic_measurements(X, max_r, max_sigma)

    params = get_default_params()
    estimator = Estimator(n, params)

    X_hat, cost_re, l_bound = estimator.estimate_RE(indices, Y, sigmas)

    X_hat_refined, cost_k = estimator.estimate_gradient(indices, Y, sigmas, X_hat, l_bound=l_bound)

    # utils.plot_unbiased(
    #     [X, X_hat, X_hat_refined],
    #     ["Real", "Estimate", "Refined"],
    #     show=True,
    # )

    return X, X_hat, X_hat_refined, indices


if __name__ == "__main__":
    plt.style.use("bmh")
    np.seterr("raise")
    run_kruskal_tests()
    # run_RE_tests()
