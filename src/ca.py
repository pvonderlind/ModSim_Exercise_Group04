from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random
from tqdm import tqdm


class Street:
    """
    Represents a street with multiple lanes and
    multiple cars that have a maximum velocity.

    Cars are represented as integers from values between
    0 and vmax. Empty cells are represented as -1.
    """

    def __init__(self,
                 lanes: int,
                 lane_len: int,
                 n_cars: int,
                 v_max: int):
        self._lanes = lanes
        self._lane_len = lane_len
        self._n_cars = n_cars
        self._v_max = v_max
        self._state = self._init_state()

    def _init_state(self) -> np.ndarray:
        velocities = [random.randint(0, self._v_max) for _ in range(self._n_cars)]
        flat_len = self._lanes * self._lane_len
        flat_street = np.full(flat_len, -1)
        cars = np.random.choice(flat_len, self._n_cars)
        flat_street[cars] = velocities
        return flat_street.reshape(self._lanes, self._lane_len)

    def update(self, new_state: np.ndarray) -> np.ndarray:
        self._state = new_state
        return self._state

    def get_state(self) -> np.ndarray:
        return self._state


class AbstractRule(ABC):
    """
    Applies a rule to all cars in the given state
    array when using the apply function.
    """

    @abstractmethod
    def apply(self, state: np.ndarray) -> np.ndarray:
        pass


class Runner:

    def __init__(self,
                 street: Street,
                 rule_list: List[AbstractRule],
                 max_timesteps: int = 250):
        self._street = street
        self._max_timesteps = max_timesteps
        self._rule_list = rule_list
        self.history = []

    def run(self, tqdm_widget=None, debug_metrics=True):
        """
        When called in the UI, to display the progress a tqdm widget displaying the live
        progress can be passed as `tqdm_widget`. It is used to loop just like tqdm().
        """
        
        if tqdm_widget:
            tqdm = tqdm_widget
            
        print(f"Starting simulation".center(50, '.'))
        for _ in tqdm(range(self._max_timesteps)):
            new_state = self._apply_rules(self._street)
            if debug_metrics:
                print(f"Current number of cars {(new_state >= 0).sum()}")
            self._street.update(new_state)
            self.history.append(new_state)
        print(f"Ended simulation after {self._max_timesteps} steps!".center(50, '.'))

    def _apply_rules(self, street: Street) -> np.ndarray:
        state = street.get_state()
        for rule in self._rule_list:
            state = rule.apply(state)
        return state
