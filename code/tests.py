import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial
import utils
from estimator import Estimator, get_default_ekf_funcs, get_default_params
from matplotlib.animation import FuncAnimation


def run_kruskal_tests_old():
    np.random.seed(578)
    ns = [4, 10]
    # ns = [50]
    ms = [[5, 11], [40, 85]]
    # ms = [[200, 500]]
    # grids = [(2, 2), (5, 4)]
    grids = [(2, 2), (5, 2)]

    for it, (n, grid) in enumerate(zip(ns, grids)):
        for m in ms[it]:
            trial_kruskal, trial_naive = kruskal_test_old(m, n, 5, *grid)

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


def kruskal_test_old(m, n, N, w, h):
    params = get_default_params()
    funcs = get_default_ekf_funcs(n, params.dt)
    estimator = Estimator(n, params, funcs)

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


def run_kruskal_tests():
    ns = [6, 10, 15]
    ms = [[9, 25], [25, 79], [50, 109]]
    dims = [[3, 2], [5, 2], [5, 3]]

    ns = [4, 10, 15]
    ms = [7, 45, 210]
    dims = [[2, 2], [5, 2], [5, 3]]

    fig, axs = plt.subplots(nrows=len(ns), sharex=True)
    if len(ns) == 1:
        axs = [axs]

    axs[-1].set_xlabel("Iterations")
    # fig, axs = plt.subplots(nrows=3)
    # axs[0].set_title("Kruskal")
    # axs[1].set_title("Inverse square root")
    # axs[2].set_title("Constant")

    for it, (n, m, dim) in enumerate(zip(ns, ms, dims)):
        w, h = dim
        axs[it].set_title(f"$({n = }, {m = })$")

        trial_kruskal, trial_sqrt_inv, trial_const = kruskal_test(
            m,
            n,
            w,
            h,
            max_its=50,
            kruskal_init=0.4,
            sqrt_inv_init=10,
            const_init=10,
        )

        axs[it].plot(range(len(trial_kruskal)), trial_kruskal, label="Kruskal", linestyle="-")
        axs[it].plot(range(len(trial_sqrt_inv)), trial_sqrt_inv, label="Inverse root", linestyle="--")
        axs[it].plot(range(len(trial_const)), trial_const, label="Constant", linestyle="-.")

        print("trial_kruskal", len(trial_kruskal))
        print("trial_sqrt", len(trial_sqrt_inv))
        print("trial_const", len(trial_const))

    #         axs[0].plot(trial_kruskal, label=f"${n = }, {m = }$")
    #         axs[1].plot(trial_sqrt_inv, label=f"${n = }, {m = }$")
    #         axs[2].plot(trial_const, label=f"${n = }, {m = }$")

    for ax in axs:
        ax.legend()
        ax.set_yscale("log")
        ax.set_ylabel("Cost")

    plt.tight_layout()
    plt.show()


def run_kruskal_tests2():
    n = 15
    m = 150
    dim = [5, 3]

    params_list = [[0.1, 10, 10], [0.2, 20, 20], [0.4, 40, 40]]
    # params_list = [[0.1, 10, 10], [0.2, 20, 20]]
    # params_list = [[0.1, 10, 10]]
    styles = ["-", "--", "-."]

    fig, axs = plt.subplots(nrows=3, sharex=False)
    axs[-1].set_xlabel("Iterations")

    for it, params in enumerate(params_list):
        w, h = dim
        axs[0].set_title(f"Kruskal's method")
        axs[1].set_title(f"Inverse square root")
        axs[2].set_title(f"Constant step size")
        np.random.seed(31415)

        # Disable cases that do not work
        if params[1] > 20:
            params[1] = 21
        if params[2] > 20:
            params[2] = 21
        trial_kruskal, trial_sqrt_inv, trial_const = kruskal_test(m, n, w, h, *params, max_its=200)

        axs[0].plot(range(len(trial_kruskal)), trial_kruskal, label=f"$\\alpha_0={params[0]}$", linestyle=styles[it])
        if params[1] <= 20:
            axs[1].plot(
                range(len(trial_sqrt_inv)),
                trial_sqrt_inv,
                label=f"$\\frac{{{params[1]}}}{{\\sqrt{{k}}}}$",
                linestyle=styles[it],
            )
        if params[2] <= 20:
            axs[2].plot(range(len(trial_const)), trial_const, label=f"$\\alpha={params[2]}$", linestyle=styles[it])

    for ax in axs:
        ax.legend()
        ax.set_yscale("log")
        ax.set_ylim([5e1, 2e3])
        ax.set_ylabel("Cost")

    plt.suptitle(f"Comparison for ${n = }, {m = }$")
    plt.tight_layout()
    plt.show()


def kruskal_test(m, n, w, h, kruskal_init=1, sqrt_inv_init=10, const_init=6, max_its=30):
    params = get_default_params()
    funcs = get_default_ekf_funcs(n, params.dt)

    estimator = Estimator(n, params, funcs)

    X = utils.generate_grid(w, h)
    X += 100 * np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
    indices = utils.generate_indices(m, n)
    sigmas = utils.generate_sigmas(m, min=25, max=75)
    Y = utils.generate_measurements(X, indices, sigmas)

    # Initial solve
    X_hat_re, _, l_bound = estimator.estimate_RE(indices, Y, sigmas, verbose=False)
    _, cost_kruskal = estimator.estimate_gradient(
        indices,
        Y,
        sigmas,
        X_hat_re,
        l_bound=l_bound,
        verbose=False,
        kruskal=True,
        max_its=max_its,
        init_step=kruskal_init,
    )
    _, cost_sqrt_inv = estimator.estimate_gradient(
        indices,
        Y,
        sigmas,
        X_hat_re,
        l_bound=l_bound,
        verbose=False,
        kruskal=False,
        sqrt_inv=True,
        init_step=sqrt_inv_init,
        max_its=max_its,
    )
    _, cost_const = estimator.estimate_gradient(
        indices,
        Y,
        sigmas,
        X_hat_re,
        l_bound=l_bound,
        verbose=False,
        kruskal=False,
        const=True,
        init_step=const_init,
        max_its=max_its,
    )

    return cost_kruskal, cost_sqrt_inv, cost_const


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
    funcs = get_default_ekf_funcs(n, params.dt)
    estimator = Estimator(n, params, funcs)

    X_hat, cost_re, l_bound = estimator.estimate_RE(indices, Y, sigmas)

    X_hat_refined, cost_k = estimator.estimate_gradient(indices, Y, sigmas, X_hat, l_bound=l_bound)

    # utils.plot_unbiased(
    #     [X, X_hat, X_hat_refined],
    #     ["Real", "Estimate", "Refined"],
    #     show=True,
    # )

    return X, X_hat, X_hat_refined, indices


def test_success_rates():
    N = 10
    ns = [6, 10, 15]
    ms = []
    # ms += [np.linspace(1 * 6, 6 * (6 - 1), 6 - (1 - 1), dtype=int)]
    ms += [np.linspace(9, 30, 11, dtype=int)]
    # ms += [np.linspace(2 * 10, 10 * (10 - 1), 10 - (2 - 1), dtype=int)]
    ms += [np.linspace(20, 90, 21, dtype=int)]
    # ms += [np.linspace(3 * 15, 15 * (15 - 1), 15 - (3 - 1), dtype=int)]
    ms += [np.linspace(45, 210, 16, dtype=int)]
    dims = [[3, 2], [5, 2], [5, 3]]

    success_rates_RE = {n: {} for n in ns}
    success_rates_good_guess = {n: {} for n in ns}
    success_rates_small_guess = {n: {} for n in ns}
    success_rates_big_guess = {n: {} for n in ns}

    times_RE = {n: {} for n in ns}
    times_good_guess = {n: {} for n in ns}
    times_small_guess = {n: {} for n in ns}
    times_big_guess = {n: {} for n in ns}

    for i, (n, dim) in enumerate(zip(ns, dims)):
        w, h = dim
        for m in ms[i]:
            print(f"{n = } \t {m = }")
            utils.tick()
            s_rate = RE_success_rate(m, n, N, w, h, verbose=False)
            t = utils.tock()
            success_rates_RE[n][int(m)] = int(s_rate * 100)
            times_RE[n][int(m)] = t / N

            generator = lambda: np.vstack([np.random.uniform(0, 1600, size=n), np.random.uniform(0, 900, size=n)]).T
            utils.tick()
            s_rate = random_success_rate(m, n, N, w, h, generator, verbose=False)
            t = utils.tock()
            success_rates_good_guess[n][int(m)] = int(s_rate * 100)
            times_good_guess[n][int(m)] = t / N

            generator = lambda: np.random.multivariate_normal([0, 0], [[10, 0], [0, 10]], size=n)
            utils.tick()
            s_rate = random_success_rate(m, n, N, w, h, generator, verbose=False)
            t = utils.tock()
            success_rates_small_guess[n][int(m)] = int(s_rate * 100)
            times_small_guess[n][int(m)] = t / N

            generator = lambda: np.random.multivariate_normal([0, 0], [[10 * 1600**2, 0], [0, 10 * 900**2]], size=n)
            utils.tick()
            s_rate = random_success_rate(m, n, N, w, h, generator, verbose=False)
            t = utils.tock()
            success_rates_big_guess[n][int(m)] = int(s_rate * 100)
            times_big_guess[n][int(m)] = t / N

    fig, axs = plt.subplots(len(ns))

    if len(ns) == 1:
        axs = [axs]

    for i, n in enumerate(ns):
        axs[i].plot(
            ms[i],
            [success_rates_RE[n][m] for m in ms[i]],
            color="#8C1515",
            marker="x",
            markersize=10,
            linestyle="-",
            label="Riemannian elevator",
        )
        axs[i].plot(
            ms[i],
            [success_rates_good_guess[n][m] for m in ms[i]],
            color="C2",
            marker="+",
            markersize=10,
            linestyle="--",
            label="Informed guess",
        )
        axs[i].plot(
            ms[i],
            [success_rates_small_guess[n][m] for m in ms[i]],
            color="C3",
            marker="+",
            markersize=10,
            linestyle="-.",
            label="Too small guess",
        )
        axs[i].plot(
            ms[i],
            [success_rates_big_guess[n][m] for m in ms[i]],
            color="C4",
            marker="v",
            markersize=10,
            linestyle=":",
            label="Too big guess",
        )
        axs[i].set_xlabel("m")
        axs[i].set_ylabel("Success rate [%]")
        axs[i].set_title(f"{n = }")
        # axs[i].legend()
    axs[0].legend()

    plt.tight_layout()

    fig, axs = plt.subplots(len(ns))

    if len(ns) == 1:
        axs = [axs]

    for i, n in enumerate(ns):
        axs[i].plot(
            ms[i],
            [times_RE[n][m] for m in ms[i]],
            color="#8C1515",
            marker="x",
            markersize=10,
            linestyle="-",
            label="Riemannian elevator",
        )
        axs[i].plot(
            ms[i],
            [times_good_guess[n][m] for m in ms[i]],
            color="C2",
            marker="+",
            markersize=10,
            linestyle="--",
            label="Informed guess",
        )
        axs[i].plot(
            ms[i],
            [times_small_guess[n][m] for m in ms[i]],
            color="C3",
            marker="v",
            markersize=10,
            linestyle="-.",
            label="Too small guess",
        )
        axs[i].plot(
            ms[i],
            [times_big_guess[n][m] for m in ms[i]],
            color="C4",
            marker="^",
            markersize=10,
            linestyle=":",
            label="Too big guess",
        )
        axs[i].legend()
        axs[i].set_xlabel("m")
        axs[i].set_ylabel("Runtime")
        axs[i].set_title(f"{n = }")

    plt.tight_layout()
    plt.show()


def RE_success_rate(m, n, N, w, h, verbose=False):
    successes = 0
    for it in range(N):
        params = get_default_params()
        funcs = get_default_ekf_funcs(n, params.dt)

        estimator = Estimator(n, params, funcs)

        X = utils.generate_grid(w, h)
        X += 100 * np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
        indices = utils.generate_indices(m, n)
        sigmas = utils.generate_sigmas(m, min=25, max=25)
        Y = utils.generate_measurements(X, indices, sigmas)

        # Initial solve
        X_hat_re, _, l_bound = estimator.estimate_RE(indices, Y, sigmas, verbose=verbose)
        X_hat_k, _ = estimator.estimate_gradient(indices, Y, sigmas, X_hat_re, l_bound=l_bound, verbose=verbose)

        X_norm, X_hat_re_norm, re_cost = scipy.spatial.procrustes(X, X_hat_re)
        X_norm, X_hat_k_norm, k_cost = scipy.spatial.procrustes(X, X_hat_k)

        if verbose:
            print(f"Iteration: {it} \t Max dist: {np.linalg.norm(X_norm - X_hat_k_norm, axis=1).max()}")

        # if k_cost < 0.01:
        if np.linalg.norm(X_norm - X_hat_k_norm, axis=1).max() < 0.1:
            successes += 1
            # print(f"{it}: Success")
        else:
            # print("Failure")
            pass

        # plt.scatter(X_norm[:, 0], X_norm[:, 1], label="Real", marker="o", s=90)
        # plt.scatter(X_hat_re_norm[:, 0], X_hat_re_norm[:, 1], label="Estimate", marker="+", s=90)
        # plt.scatter(X_hat_k_norm[:, 0], X_hat_k_norm[:, 1], label="Refined", marker="x", s=90)
        # plt.legend()
        # plt.show()

    return successes / N


def kalman_test_map(m, n, N, w, h):
    # Initialize
    params = get_default_params()
    params.dt = 0.5
    params.kalman_R_gain *= 1
    funcs = get_default_ekf_funcs(n, params.dt)

    estimator = Estimator(n, params, funcs)

    X = utils.generate_grid(w, h)
    X += 100 * np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
    thetas = np.random.uniform(0, 2 * np.pi, size=(n, 1))
    states = np.hstack((X, thetas))

    indices = utils.generate_indices(m, n)
    # sigmas = 10 * np.ones(m)
    sigmas = utils.generate_sigmas(m, min=25, max=25)
    Y = utils.generate_measurements(X, indices, sigmas)

    # Initial solve
    X_hat, cost_RE, l_bound = estimator.estimate_RE(indices, Y, sigmas)
    X_hat, cost_K = estimator.estimate_gradient(indices, Y, sigmas, X_hat, l_bound=l_bound)

    utils.plot_unbiased(plt.gca(), [X, X_hat], ["Real", "Estimate"], ["o", "x"], [40, 80], ["C1", "C2"])
    plt.legend()
    plt.show()

    estimator.set_priors(X_hat, np.log(cost_K[-1]))

    # Simulate & estimate
    u = np.zeros(2 * n)
    # dt = 0.1
    # u[::2] = np.abs(np.random.uniform(50, 200, size=n))
    # u[1::2] = np.random.choice(np.array([-1, 1]), n) * np.random.uniform(0.2, 0.7)
    u[::2] = np.abs(np.random.uniform(5, 30, size=n))
    # u[1::2] = np.random.choice(np.array([-1, 1]), n) * np.random.uniform(0.01, 0.2)

    state_hist = np.zeros((n, 3, N))
    pred_hist = np.zeros((n, 3, N))
    meas_hist = np.zeros((n, 2, N))
    est_hist = np.zeros((n, 3, N))

    for i in range(N):
        # Where do we think the points will be
        state_pred = estimator.ekf_predict(u)
        pred_hist[:, :, i] = state_pred

        # Update robots
        states[:, :] = funcs.f(states.reshape(-1), u).reshape(-1, 3)
        state_hist[:, :, i] = states

        indices = utils.generate_indices(m, n)
        Y = utils.generate_measurements(states[:, :2], indices, sigmas)

        # Add measurement
        X_hat, cost = estimator.estimate_gradient(indices, Y, sigmas, state_pred[:, :2], eps=m * 1e-5)
        meas_hist[:, :, i] = X_hat

        state_est = estimator.ekf_update(X_hat)
        est_hist[:, :, i] = state_est

    fig = plt.gcf()
    ax = plt.gca()

    state_pts = ax.scatter(
        state_hist[:, 0, 0],
        state_hist[:, 1, 0],
        label="Real",
        marker="o",
        s=100,
        linewidth=1,
        edgecolors="C1",
        facecolors="none",
    )
    pred_pts = ax.scatter(pred_hist[:, 0, 0], pred_hist[:, 1, 0], s=100, label="Prediction", marker="+", c="C2")
    meas_pts = ax.scatter(meas_hist[:, 0, 0], meas_hist[:, 1, 0], s=100, label="Measurement", marker="x", c="C3")
    est_pts = ax.scatter(
        est_hist[:, 0, 0],
        est_hist[:, 1, 0],
        label="Estimation",
        marker="s",
        s=70,
        linewidth=2,
        edgecolors="C4",
        facecolors="none",
    )

    last_state = np.zeros_like(state_hist[:, :2, 0])
    last_pred = np.zeros_like(pred_hist[:, :2, 0])
    last_meas = np.zeros_like(meas_hist[:, :2, 0])
    last_est = np.zeros_like(est_hist[:, :2, 0])

    def update(frame, histories, lasts):
        if frame == 1:
            print("Restarting...")
        if (frame + 1) % 50 == 0:
            print(f"Drawing frame {frame+1}...")

        state_hist, pred_hist, meas_hist, est_hist = histories
        last_state, last_pred, last_meas, last_est = lasts

        frame_idx = frame // 3

        # state = utils.get_unbiased_coords(state_hist[:, :2, frame_idx], last_state)
        # pred = utils.get_unbiased_coords(pred_hist[:, :2, frame_idx], last_state)
        # meas = utils.get_unbiased_coords(meas_hist[:, :2, frame_idx], last_state)
        # est = utils.get_unbiased_coords(est_hist[:, :2, frame_idx], last_state)
        # state = state_hist[:, :2, frame_idx]
        # pred = pred_hist[:, :2, frame_idx]
        # meas = meas_hist[:, :2, frame_idx]
        # est = est_hist[:, :2, frame_idx]
        state, pred, meas, est = utils.get_aligned_coords(
            state_hist[:, :2, frame_idx],
            [pred_hist[:, :2, frame_idx], meas_hist[:, :2, frame_idx], est_hist[:, :2, frame_idx]],
        )

        if frame % 4 >= 0:
            state_pts.set_offsets(state)
        else:
            state_pts.set_offsets(np.empty((0, 2)))

        if frame % 4 >= 1:
            pred_pts.set_offsets(pred)
        else:
            pred_pts.set_offsets(np.empty((0, 2)))

        if frame % 4 >= 2:
            meas_pts.set_offsets(meas)
        else:
            meas_pts.set_offsets(np.empty((0, 2)))

        if frame % 4 >= 3:
            est_pts.set_offsets(est)
        else:
            est_pts.set_offsets(np.empty((0, 2)))

        # state_pts.set_offsets(state)
        # pred_pts.set_offsets(pred)

        lasts[0] = state
        lasts[1] = pred
        lasts[2] = meas
        lasts[3] = est

        return [state_pts, pred_pts, meas_pts, est_pts]

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")

    ani = FuncAnimation(
        fig,
        update,
        frames=np.arange(3 * N),
        blit=True,
        fargs=((state_hist, pred_hist, meas_hist, est_hist), [last_state, last_pred, last_meas, last_est]),
        interval=1000,
        repeat=True,
    )  # type: ignore
    plt.legend()

    # ani.save("animation.mp4", writer="ffmpeg")
    plt.show()
    # utils.plot_points([X, X_hat], ["X", "X_hat"], ["x", "+"], [50, 50], ["C1", "C2"], plt.gca())

    # plt.legend()
    # plt.show()


def kalman_test_plot(m, n, N, w, h, angle_params):
    # Initialize
    # matplotlib.rcParams["font.family"] = "serif"
    params = get_default_params()
    params.kalman_R_gain *= 1
    params.dt = 2
    funcs = get_default_ekf_funcs(n, params.dt)

    estimator = Estimator(n, params, funcs)

    X = utils.generate_grid(w, h)
    X += 100 * np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
    thetas = np.random.uniform(0, 2 * np.pi, size=(n, 1))
    states = np.hstack((X, thetas))

    indices = utils.generate_indices(m, n)
    sigmas = utils.generate_sigmas(m, min=200, max=400)
    Y = utils.generate_measurements(X, indices, sigmas)

    # Initial solve
    X_hat, cost_RE, l_bound = estimator.estimate_RE(indices, Y, sigmas)
    X_hat, cost_K = estimator.estimate_gradient(indices, Y, sigmas, X_hat, l_bound=l_bound)

    utils.plot_unbiased(plt.gca(), [X, X_hat], ["Real", "Estimate"], ["o", "x"], [40, 80], ["C1", "C2"])
    plt.legend()
    # plt.show()

    estimator.set_priors(X_hat, np.log(cost_K[-1]))

    # Simulate & estimate
    u = np.zeros(2 * n)
    us = np.zeros((2 * n, N))
    # dt = 0.1
    # u[::2] = np.abs(np.random.uniform(50, 200, size=n))
    # u[1::2] = np.random.choice(np.array([-1, 1]), n) * np.random.uniform(0.2, 0.7)
    u[::2] = np.abs(np.random.uniform(10, 30, size=n))
    u[1::2] = np.random.uniform(-0.0, 0.00)

    u_b1 = np.abs(np.random.uniform(10, 30, size=(n, N // 5)))
    u_b2 = np.random.normal(0, 0.1, size=(n, N // 5))
    us[::2] = np.repeat(u_b1, 5, axis=1)
    us[1::2] = np.repeat(u_b2, 5, axis=1)

    vdot = np.random.normal(0, 1, size=(n, N))
    wdot = np.random.normal(0, 0.01, size=(n, N))

    vdot = np.random.normal(0, 3, size=(n, N // 5))
    wdot = np.random.normal(0, 0.1, size=(n, N // 5))

    vdot = np.repeat(vdot, 5, axis=1)
    wdot = np.repeat(wdot, 5, axis=1)

    state_hist = np.zeros((n, 3, N))
    pred_hist = np.zeros((n, 3, N))
    meas_hist = np.zeros((n, 2, N))
    est_hist = np.zeros((n, 3, N))

    for i in range(N):
        u = us[:, i]
        # Where do we think the points will be
        state_pred = estimator.ekf_predict(u)
        pred_hist[:, :, i] = state_pred

        # Update robots
        states[:, :] = funcs.f(states.reshape(-1), u).reshape(-1, 3)
        state_hist[:, :, i] = states

        indices = utils.generate_indices(m, n)
        Y = utils.generate_measurements(states[:, :2], indices, sigmas)

        # Add measurement
        X_hat, cost = estimator.estimate_gradient(indices, Y, sigmas, state_pred[:, :2], eps=m * 1e-5)
        meas_hist[:, :, i] = X_hat

        state_est = estimator.ekf_update(X_hat)
        est_hist[:, :, i] = state_est

        # u[::2] *= np.random.uniform(1 / 0.95, 0.98, size=n)
        # u[1::2] *= np.random.uniform(1 / 0.98, 0.95, size=n)

        # u[::2] += vdot[:, i]
        # u[::2] = np.maximum(u[::2], 0)
        # u[1::2] += wdot[:, i]

    state_std = np.zeros_like(state_hist)
    est_std = np.zeros_like(est_hist)

    # Plot position over time
    for i in range(N):
        state_std[:, :2, i], est_std[:, :2, i], loss = scipy.spatial.procrustes(
            state_hist[:, :2, i], est_hist[:, :2, i]
        )

    sign, offset, wrap = angle_params
    if wrap:
        state_std[:, 2, :] = state_hist[:, 2, :] % (2 * np.pi)
        est_std[:, 2, :] = (sign * est_hist[:, 2, :] + offset) % (2 * np.pi)
        # state_std[:, 2, :] = np.cos(state_hist[:, 2, :])
        # est_std[:, 2, :] = np.cos(sign * est_hist[:, 2, :] + offset)
    else:
        state_std[:, 2, :] = state_hist[:, 2, :]
        est_std[:, 2, :] = sign * est_hist[:, 2, :] + offset

    fig, axs = plt.subplots(n // 4, 3, sharex=True)

    axs[-1, 0].set_xlabel("Iteration")
    axs[-1, 1].set_xlabel("Iteration")
    axs[-1, 2].set_xlabel("Iteration")
    for k in range(n // 4):
        axs[k, 0].plot(state_std[k, 0, :], label="Real", linestyle="-", linewidth=4)
        axs[k, 0].plot(est_std[k, 0, :], label="Est", linestyle="--")
        # axs[k, 0].scatter(range(N), est_std[k, 0, :], label="Est", marker="x", c="C4")
        # axs[k, 0].plot(np.linalg.norm(est_std[k, :2, :] - state_std[k, :2, :], axis=0))
        # axs[k, 0].set_title("$||r_{pos}||^2_2$")
        axs[k, 0].set_title(f"Robot {k+1} $x$")
        axs[k, 0].set_ylabel("Position")
        # axs[k, 0].legend()

        axs[k, 1].plot(state_std[k, 1, :], label="Real", linestyle="-", linewidth=4)
        axs[k, 1].plot(est_std[k, 1, :], label="Est", linestyle="--")
        # axs[k, 1].scatter(range(N), est_std[k, 1, :], label="Est", marker="x", c="C4")
        axs[k, 1].set_title(f"Robot {k+1} $y$")
        axs[k, 1].set_ylabel("Position")
        # axs[k, 1].legend()

        axs[k, 2].plot(state_std[k, 2, :], label="Real", linestyle="-", linewidth=4)
        axs[k, 2].plot(est_std[k, 2, :], label="Est", linestyle="--")
        # axs[k, 2].scatter(range(N), est_std[k, 2, :], label="Est", marker="x", c="C4")
        axs[k, 2].set_title(f"Robot {k+1} $\\theta$")
        axs[k, 2].set_ylabel("Angle [rad]")
        # axs[k, 2].legend()

    plt.tight_layout()
    # plt.suptitle(f"Robot tracking ($m = {m}$, $n = {n}$, $\\sigma={sigmas[-1]})$")
    plt.show()


def run_random_solve_test():
    # N = 20
    # ns = [4, 7, 10]
    # ms = []
    # # ms += [np.linspace(9, 30, 11, dtype=int)]
    # ms += [np.linspace(4, 12, 9, dtype=int)]
    # ms += [np.linspace(10, 35, 13, dtype=int)]
    # ms += [np.linspace(31, 90, 15, dtype=int)]
    # # dims = [[3, 2], [5, 2], [5, 3]]
    # dims = [[2, 2], [7, 1], [5, 2]]

    # Ns_needed = {n: {} for n in ns}

    # for i, (n, dim) in enumerate(zip(ns, dims)):
    #     w, h = dim
    #     for m in ms[i]:
    #         print(f"{n = } \t {m = }")
    #         X = utils.generate_random_positions(n)
    #         indices = utils.generate_indices(m, n)
    #         sigmas = utils.generate_sigmas(m, 1, 10)
    #         Y = utils.generate_measurements(X, indices, sigmas)

    #         _, N_needed = solve_randomly(X, Y, indices, sigmas, N)

    #         Ns_needed[n][int(m)] = N_needed

    # print(json.dumps(Ns_needed, indent=2))

    N = 50
    ns = [6, 10, 15]
    ms = []
    # ms += [np.linspace(1 * 6, 6 * (6 - 1), 6 - (1 - 1), dtype=int)]
    ms += [np.linspace(9, 30, 11, dtype=int)]
    # ms += [np.linspace(2 * 10, 10 * (10 - 1), 10 - (2 - 1), dtype=int)]
    ms += [np.linspace(20, 90, 21, dtype=int)]
    # ms += [np.linspace(3 * 15, 15 * (15 - 1), 15 - (3 - 1), dtype=int)]
    ms += [np.linspace(45, 210, 16, dtype=int)]
    dims = [[3, 2], [5, 2], [5, 3]]

    success_rates = {n: {} for n in ns}

    for i, (n, dim) in enumerate(zip(ns, dims)):
        w, h = dim
        for m in ms[i]:
            print(f"{n = } \t {m = }")
            s_rate = random_success_rate(m, n, N, w, h, verbose=False)
            success_rates[n][int(m)] = int(s_rate * 100)

    print(json.dumps(success_rates, indent=2))


def random_success_rate(m, n, N, w, h, guess_generator, verbose):
    successes = 0
    for it in range(N):
        params = get_default_params()
        funcs = get_default_ekf_funcs(n, params.dt)

        estimator = Estimator(n, params, funcs)

        X = utils.generate_grid(w, h)
        X += 100 * np.random.multivariate_normal(np.zeros(2), np.eye(2), size=n)
        indices = utils.generate_indices(m, n)
        sigmas = utils.generate_sigmas(m, min=25, max=25)
        Y = utils.generate_measurements(X, indices, sigmas)

        # X_0 = np.vstack([np.random.uniform(0, 1600, size=n), np.random.uniform(0, 900, size=n)]).T
        X_0 = guess_generator()

        X_hat, _ = estimator.estimate_gradient(indices, Y, sigmas, X_0, verbose=verbose)

        X_norm, X_hat_norm, _ = scipy.spatial.procrustes(X, X_hat)

        if verbose:
            print(f"Iteration: {it} \t Max dist: {np.linalg.norm(X_norm - X_hat_norm, axis=1).max()}")

        # if k_cost < 0.01:
        if np.linalg.norm(X_norm - X_hat_norm, axis=1).max() < 0.1:
            successes += 1
            # print(f"{it}: Success")
        else:
            # print("Failure")
            pass

        # plt.scatter(X_norm[:, 0], X_norm[:, 1], label="Real", marker="o", s=90)
        # plt.scatter(X_hat_re_norm[:, 0], X_hat_re_norm[:, 1], label="Estimate", marker="+", s=90)
        # plt.scatter(X_hat_k_norm[:, 0], X_hat_k_norm[:, 1], label="Refined", marker="x", s=90)
        # plt.legend()
        # plt.show()

    return successes / N


# def solve_randomly(X, measurements, indices, sigmas):
#     d_hat = measurements
#     n = X.shape[0]
#     m = d_hat.shape[0]

#     params = get_default_params()
#     funcs = get_default_ekf_funcs(n, params.dt)
#     estimator = Estimator(n, params, funcs)

#     N_needed = N

#     X_0 = np.vstack([np.random.uniform(0, 1600, size=n), np.random.uniform(0, 900, size=n)]).T

#     X_hat, costs, Xs = estimator.estimate_gradient(indices, measurements, sigmas, X_0)

#     X_norm, X_hat_norm, _ = scipy.spatial.procrustes(X, X_hat)

#     # X_tilde, _, _ = estimator.estimate_RE(indices, measurements, sigmas)
#     # X_tilde, _, _ = estimator.estimate_gradient(indices, measurements, sigmas, X_tilde, verbose=False, eps=1e-6)
#     # X_norm, X_tilde_norm, _ = scipy.spatial.procrustes(X, X_tilde)
#     # plt.scatter(X_norm[:, 0], X_norm[:, 1], marker="x", label="X")
#     # plt.scatter(X_tilde_norm[:, 0], X_tilde_norm[:, 1], marker="+", label="X_tilde")
#     # plt.legend()
#     # plt.show()

#     return X_hat


if __name__ == "__main__":
    plt.style.use("bmh")
    np.seterr("raise")
    np.set_printoptions(precision=3, suppress=True)
    # np.random.seed(314112)
    # run_kruskal_tests()
    # run_RE_tests()
    # np.random.seed(3141)
    # kalman_test_plot(15 * 7, 15, 100, 5, 3, (1, 1.5 * np.pi, True))
    # kalman_test_map(12, 4, 100, 2, 2)
    # np.random.seed(3132)
    # kalman_test_plot(20, , 100, 3, 2, (+1, 1 * np.pi))

    # To generate plots:
    # np.random.seed(42)
    # kalman_test_plot(15 * 8, 15, 50, 5, 3, (-1, 0.3 * np.pi, True))
    # kalman_test_plot(120, 15, 100, 5, 3, (1, 1 * np.pi, True))
    np.random.seed(12323)
    kalman_test_plot(120, 15, 100, 5, 3, (1, 2 * np.pi, True))

    # To generate plots for poster:
    # np.random.seed(31415)
    # test_success_rates()

    # More poster plots:
    # np.random.seed(12314)
    # run_kruskal_tests()

    # np.random.seed(31415)
    # run_random_solve_test()

    # run_kruskal_tests2()

# Up to 15 targets work (maybe more?), depending on noise and number of connections
