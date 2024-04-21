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
            gaussian,
            agent.CirclePath(0.2, 40, offset=10 * i),
            # agent.StillPath(),
        )
        agents.append(ag)
    root, canvas = gui.init(agents)

    worker = Thread(target=update_agent, args=(agents,), name="worker", daemon=True)
    worker.start()

    gui.start(root)


def update_agent(agents):
    while True:
        for ag in agents:
            ag.move()
            with ag.get_lock():
                pos = ag.get_pos()
            sleep(Params.move_delay)


if __name__ == "__main__":
    main()
