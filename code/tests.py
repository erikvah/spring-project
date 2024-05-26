import matplotlib.pyplot as plt
import numpy as np
import utils
from estimator import Estimator, get_default_params


def run_kruskal_tests():
    ns = [4, 20]
    ns = [50]
    # ms = [[6, 12], [200, 380]]
    ms = [[200, 500]]
    # grids = [(2, 2), (5, 4)]
    grids = [(10, 5)]

    # plt.style.use("fivethirtyeight")
    plt.style.use("bmh")

    # ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

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
            ax_k.legend(["Kruskal cost", "Robots move"])
            ax_k.set_yscale("log")
            plt.show()

    # for cost_hist in cost_hists:
    #     plt.plot()


def kruskal_test(m, n, N, w, h):
    np.random.seed(m * n)
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

    for it in range(N):
        X += np.random.multivariate_normal(np.zeros(2), 100 * np.eye(2), size=n)
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

    # return [item for sublist in two_d_list for item in sublist]]

    return cost_hists_kruskal, cost_hists_naive


if __name__ == "__main__":
    run_kruskal_tests()
    # kruskal_test(12, 4, 5, 2, 2)
