from threading import Thread
from time import sleep, time

import agent
import gui
import numpy as np
from noise import gaussian
from params import Params


def main():
    agents = []
    n_agents = 40
    np.random.seed(0)
    x = np.random.uniform(Params.margin, Params.window_size[0] - 100, n_agents)
    y = np.random.uniform(Params.margin, Params.window_size[1] - Params.margin, n_agents)
    X = np.vstack((x, y)).T

    for i in range(n_agents):
        ag = agent.Agent(
            gaussian(0, 1),
            agent.PeriodicPath(X[i, :], 1 + int((i + 1) / 4), 0.004),
            # agent.PeriodicPath(np.array([300 + 20 * i, 500]), 1 + int((i + 1) / 4), 0.003),
            # agent.CirclePath(0.02, 30 + 10 * i, offset=10 * i),
            # agent.StillPath(),
        )
        agents.append(ag)

    for i in range(4):
        agents[4].add_meas_target(agents[i])

    root, canvas = gui.init(agents)

    worker = Thread(target=update_agents, args=(agents,), name="worker", daemon=True)
    worker.start()

    gui.start(root)


def update_agents(agents: list[agent.Agent]):
    while True:
        t0 = time()
        for ag in agents:
            ag.move()
            ag.measure_dists()
        time_sleep = max(Params.move_delay - (time() - t0), 0)
        sleep(time_sleep)
        # print("Sleeping", int(1000 * time_sleep), "ms")


if __name__ == "__main__":
    main()
