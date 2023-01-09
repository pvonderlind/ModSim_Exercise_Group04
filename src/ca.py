from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random
from tqdm import tqdm
import io
import pickle

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

    def run(self, tqdm_widget=None):
        """
        When called in the UI, to display the progress a tqdm widget displaying the live
        progress can be passed as `tqdm_widget`. It is used to loop just like tqdm().
        """
        
        if tqdm_widget:
            tqdm = tqdm_widget
            
        print(f"Starting simulation".center(50, '.'))
        for _ in tqdm(range(self._max_timesteps)):
            new_state = self._apply_rules(self._street)
            self._street.update(new_state)
            self.history.append(new_state)
        print(f"Ended simulation after {self._max_timesteps} steps!".center(50, '.'))

    def _apply_rules(self, street: Street) -> np.ndarray:
        state = street.get_state()
        for rule in self._rule_list:
            state = rule.apply(state)
        return state
    
    def serialize(self) -> bytes:
        '''
        Serializes the runner object to bytes.
        '''        
        # create in-memory file-like object
        history_compressed = io.BytesIO()
        # save numpy array to file-like object
        np.savez_compressed(history_compressed, history=np.array(self.history))
        # reset the file pointer to the beginning, so that the file can be read
        # otherwise the next read would start at the end of the file
        # and the file would be seen as empty or corrupted
        history_compressed.seek(0)
        
        # keep only the parameters of the street, not the state
        street_parameters = self._street.__dict__
        street_parameters.pop('_state')
        
        # keep the list of rules
        rule_list = self._rule_list
        
        serialized_runner = pickle.dumps({
            'street_parameters': street_parameters,
            'rule_list': rule_list,
            'history_compressed': history_compressed
        })
        
        return serialized_runner
    
    @classmethod
    def deserialize(cls, serialized_runner: bytes) -> 'Runner':
        '''
        Deserializes the runner object from bytes.
        '''
        # load the serialized runner
        serialized_runner = pickle.loads(serialized_runner)
        
        # create a new street
        sp = serialized_runner['street_parameters']
        street = Street(sp['_lanes'], sp['_lane_len'], sp['_n_cars'], sp['_v_max'])
        
        # load the history
        with np.load(serialized_runner['history_compressed']) as data:
            history_ndarray: np.ndarray = data['history']
        
        # create a new runner
        runner = cls(street, serialized_runner['rule_list'])
        runner.history = list(history_ndarray)
        
        return runner
