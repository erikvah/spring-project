from threading import Lock
from typing import Callable

import numpy as np


class _Path:
    def __init__(self):
        self._k: int = 0

    def move(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
        raise NotImplementedError()


class StillPath(_Path):
    def __init__(self):
        super().__init__()

    def move(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
        return


class SplinePath(_Path):
    def __init__(self, points):
        super().__init__()
        raise NotImplementedError()


class CirclePath(_Path):
    def __init__(self, speed: float, size: float, offset: int = 0):
        """Returns a circular path with given parameters."""
        super().__init__()
        self._size: float = size
        self._speed: float = speed
        self._k += offset

    def move(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
        new_vel_x = self._size * self._speed * np.cos(self._speed * self._k)
        new_vel_y = self._size * self._speed * np.sin(self._speed * self._k)
        acc[0] = new_vel_x - vel[0]
        acc[1] = new_vel_y - vel[1]
        vel[0] = new_vel_x
        vel[1] = new_vel_y
        pos += vel
        self._k += 1


class Agent:
    def __init__(
        self,
        init_pos: np.ndarray,
        init_vel: np.ndarray,
        init_acc: np.ndarray,
        noise: Callable,
        path: _Path,
        dim: int = 2,
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
        assert dim == 2

        self._pos = np.zeros(dim, dtype=float)
        self._vel = np.zeros(dim, dtype=float)
        self._acc = np.zeros(dim, dtype=float)

        self._pos[:] = init_pos
        self._vel[:] = init_vel
        self._acc[:] = init_acc
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
            self._path.move(self._pos, self._vel, self._acc)

    def get_measurements(self):
        """Returns copy of measurements. Not thread safe"""
        return self._measurements.copy()

    def get_pos(self):
        """Return copy of position. Not thread safe."""
        return self._pos.copy()

    def get_vel(self):
        """Return copy of velocity. Not thread safe."""
        return self._vel.copy()

    def get_acc(self):
        """Return copy of acceleration. Not thread safe."""
        return self._acc.copy()

    def get_lock(self):
        """Returns lock. Acquire before accessing
        any state field."""
        return self._lock
