import estimator
import numpy as np
import scipy.sparse as sp
import utils
from numpy.linalg import norm


class AgentHandler:
    def __init__(self, initial_points, agent_params, n):
        # Agent params:
        #   For each agent: (a1, a2, a3, ω1, ω2, ω3, sigma, range)
        self._init_positions = initial_points
        self._positions = initial_points.copy()
        self._agent_params = agent_params

        self._n = n
        self._t = 0

        assert self._positions.shape == (n, 2)
        assert self._agent_params.shape == (n, 8)

    def move_agents(self):
        self._t += 1

        A = self._agent_params[:, :3]
        Ω = self._agent_params[:, 3:6]
        self._positions[:, 0] = self._init_positions[:, 0] + (A * np.cos(Ω * self._t)) @ np.ones(
            self._n
        )
        self._positions[:, 1] = self._init_positions[:, 1] + (A * np.sin(Ω * self._t)) @ np.ones(
            self._n
        )

    def get_measurements(self):
        indices = []
        sigmas = []
        for i in range(self._n):
            norms = norm(self._positions[i, :] - self._positions, axis=1)

            for j in range(self._n):
                if norms[j] < self._agent_params[i, 7] and i != j:
                    indices.append([i, j])
                    sigmas.append(self._agent_params[i, 6])

        indices = np.array(indices)
        sigmas = np.array(sigmas)

        assert indices.size > 0

        measurements = utils.generate_measurements(self._positions, indices, sigmas)

        return indices, measurements, sigmas

    def get_positions(self):
        return self._positions


if __name__ == "__main__":
    np.random.seed(10)
    n = 50
    points = utils.generate_positions(n)
    params = np.zeros((n, 8))
    params[:, 0] = 1
    params[:, 3] = np.sqrt(np.arange(n))
    params[:, 6] = np.ones(n)
    params[:, 7] = 500
    handler = AgentHandler(points, params, n)

    alpha = 0.01
    threshold_frac = 0.02
    cholesky_noise = 0.001

    e_params = estimator.Params(alpha, threshold_frac, cholesky_noise)

    est = estimator.Estimator(n, e_params)

    indices, measurements, sigmas = handler.get_measurements()

    m = indices.shape[0]

    print(f"{n = }, {m = }")

    # X_hat = est.estimate_RE_mod(indices, measurements, sigmas)
    X_hat = est.estimate_RE(indices, measurements, sigmas)

    X = handler.get_positions()

    utils.plot_and_view_unbiased([X, X_hat], ["Real", "Estimated"], flip_y=[False, False])
