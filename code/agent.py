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
    ):

        self._pos: np.ndarray = init_pos
        self._vel: np.ndarray = init_vel
        self._acc: np.ndarray = init_acc
        self._noise: Callable = noise
        self._path: _Path = path
        self._lock: Lock = Lock()

    def move(self):
        with self._lock:
            self._path.move(self._pos, self._vel, self._acc)

    def get_pos(self):
        return self._pos.copy()

    def get_vel(self):
        return self._vel.copy()

    def get_acc(self):
        return self._acc.copy()

    def get_lock(self):
        return self._lock
