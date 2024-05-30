import matplotlib.pyplot as plt
import numpy as np
import utils
from estimator import Estimator, get_default_ekf_funcs, get_default_params
from matplotlib.animation import FuncAnimation


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


def kalman_test(m, n, N, w, h):
    # Initialize
    np.random.seed(31415)
    params = get_default_params()
    params.dt = 2
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
    u[1::2] = np.random.choice(np.array([-1, 1]), n) * np.random.uniform(0.01, 0.2)

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
        X_hat, cost = estimator.estimate_gradient(indices, Y, sigmas, state_pred[:, :2])
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

        frame_idx = frame // 4

        state = utils.get_unbiased_coords(state_hist[:, :2, frame_idx], last_state)
        pred = utils.get_unbiased_coords(pred_hist[:, :2, frame_idx], last_state)
        meas = utils.get_unbiased_coords(meas_hist[:, :2, frame_idx], last_state)
        est = utils.get_unbiased_coords(est_hist[:, :2, frame_idx], last_state)

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

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-500, 500)
    ax.set_aspect("equal")

    ani = FuncAnimation(
        fig,
        update,
        frames=np.arange(4 * N),
        blit=True,
        fargs=((state_hist, pred_hist, meas_hist, est_hist), [last_state, last_pred, last_meas, last_est]),
        interval=1000,
        repeat=True,
    )  # type: ignore
    plt.legend()

    ani.save("animation.mp4", writer="ffmpeg")
    plt.show()
    # utils.plot_points([X, X_hat], ["X", "X_hat"], ["x", "+"], [50, 50], ["C1", "C2"], plt.gca())

    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    plt.style.use("bmh")
    np.seterr("raise")
    np.set_printoptions(precision=3, suppress=True)
    # run_kruskal_tests()
    # run_RE_tests()
    np.random.seed(3141)
    kalman_test(15 * 7, 15, 10, 5, 3)
