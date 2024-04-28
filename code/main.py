from threading import Thread
from time import sleep

import agent
import gui
import numpy as np
from noise import gaussian
from params import Params


def main():
    agents = []

    for i in range(5):
        ag = agent.Agent(
            np.array([200 + 100 * i, 200]),
            np.array([0, 0]),
            np.array([0, 0]),
            gaussian(0, 1),
            agent.CirclePath(0.02, 30 + 10 * i, offset=10 * i),
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
        for ag in agents:
            ag.move()
            ag.measure_dists()

        sleep(Params.move_delay)


if __name__ == "__main__":
    main()
