import numpy as np

from src.ca import AbstractRule


class DummyShuffleRule(AbstractRule):
    def apply(self, state: np.ndarray) -> np.ndarray:
        np.random.shuffle(state)
        return state
