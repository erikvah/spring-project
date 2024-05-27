from threading import Lock
from typing import Callable

import numpy as np


class _Path:
    def __init__(self, start: np.ndarray):
        self._k: int = 0
        self._start = start
        self._pos = start

    def move(self) -> np.ndarray:
        raise NotImplementedError()

    def get_start(self) -> np.ndarray:
        return self._start.copy()


class StillPath(_Path):
    def __init__(self, start: np.ndarray):
        super().__init__(start)

    def move(self) -> np.ndarray:
        return self._pos


class SplinePath(_Path):
    def __init__(self, start: np.ndarray):
        super().__init__(start)
        raise NotImplementedError()

    def move(self) -> np.ndarray:
        raise NotImplementedError()


class PeriodicPath(_Path):
    def __init__(self, start: np.ndarray, N: int, speed: float):
        super().__init__(start)

        assert N > 0

        self._N = N
        self._speed = speed

        self._a0 = self._start[0]
        self._c0 = self._start[1]

        self._a = np.random.normal(0, 100 / np.sqrt(N), N)
        self._b = np.random.normal(0, 100 / np.sqrt(N), N)
        self._c = np.random.normal(0, 100 / np.sqrt(N), N)
        self._d = np.random.normal(0, 100 / np.sqrt(N), N)

        self._k += np.random.randint(0, 500)

    def move(self) -> np.ndarray:
        v_cos = np.array(
            [np.cos(2 * np.pi * i * self._k * self._speed / self._N) for i in range(1, self._N + 1)]
        )
        v_sin = np.array(
            [np.sin(2 * np.pi * i * self._k * self._speed / self._N) for i in range(1, self._N + 1)]
        )
        self._pos[0] = self._a0 + (v_cos * self._a + v_sin * self._b).sum()
        self._pos[1] = self._c0 + (v_cos * self._c + v_sin * self._d).sum()

        self._k += 1

        return self._pos.copy()


class CirclePath(_Path):
    def __init__(self, start: np.ndarray, speed: float, size: float, offset: int = 0):
        """Returns a circular path with given parameters."""
        super().__init__(start)
        self._size: float = size
        self._speed: float = speed
        self._k += offset

    def move(self) -> np.ndarray:
        self._pos[0] = self._start + self._size * np.cos(self._speed * self._k)
        self._pos[1] = self._start + self._size * np.sin(self._speed * self._k)

        self._k += 1

        return self._pos.copy()


class Agent:
    def __init__(
        self,
        noise: Callable,
        path: _Path,
    ):
        """An agent is an abstraction of a robot
        moving along some given path and taking noisy
        distance measurements of other robots.

        Initial state likely partially overridden
        by set movement along path.

        Arguments:
        - init_pos: Initial position
        - init_vel: Initial velocity
        - init_acc: Initial acceleration
        - noise: Noise generating function
        - path: The path to follow
        - dim: The dimensions of the position
        """
        self._pos = path.get_start()
        self._noise: Callable = noise
        self._path: _Path = path
        self._lock: Lock = Lock()
        self._meas_targets = []
        self._measurements = np.array([])

    def add_meas_target(self, target: "Agent"):  # bruh
        """Adds the given agent to the list of agents
        that this agent measures distance to.

        Arguments:
        - target: The agent to add"""
        self._meas_targets.append(target)
        self._measurements = np.zeros(len(self._meas_targets))

    def measure_dists(self):
        """Updates the measurements of the distance to
        the agents in the _meas_targets list. Thread safe."""
        with self._lock:
            for i, target in enumerate(self._meas_targets):
                with target.get_lock():
                    self._measurements[i] = (
                        np.linalg.norm(self._pos - target.get_pos()) + self._noise()
                    )

    def move(self):
        """Move agent along path. Thread safe. Updates all state fields."""
        with self._lock:
            self._pos = self._path.move()

    def get_measurements(self):
        """Returns copy of measurements. Not thread safe"""
        return self._measurements.copy()

    def get_pos(self):
        """Return copy of position. Not thread safe."""
        return self._pos.copy()

    def get_lock(self):
        """Returns lock. Acquire before accessing
        any state field."""
        return self._lock
