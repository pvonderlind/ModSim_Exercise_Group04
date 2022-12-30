import numpy as np

from ca import AbstractRule


class DummyShuffleRule(AbstractRule):
    def apply(self, state: np.ndarray) -> np.ndarray:
        return np.roll(state, 1, axis=1)
