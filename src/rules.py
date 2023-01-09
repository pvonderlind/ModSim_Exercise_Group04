import numpy as np

from src.ca import AbstractRule


class DummyShuffleRule(AbstractRule):
    def apply(self, state: np.ndarray) -> np.ndarray:
        return np.roll(state, 1, axis=1)


class Accelerate(AbstractRule):
    """
    increase vehicle speed if maximum is not yet reached
    """

    def __init__(self, v_max: int):
        self.v_max = v_max

    def apply(self, state: np.ndarray) -> np.ndarray:
        check_velocity = (state < self.v_max) & (state >= 0)
        state[check_velocity] += 1
        return state


class AvoidCollision(AbstractRule):
    """
    reduce vehicle speed to gap size, 
    if gap to next vehicle is smaller than its speed
    """

    def check_following_vehicles(self, lane: np.ndarray, index: int, speed: int) -> np.ndarray:
        """
        returns array of which following cells contain a vehicle
        """
        # move current vehicle to front
        shifted_state = np.roll(lane, -(index + 1))
        gap_ahead_of_car = shifted_state[:speed]
        return gap_ahead_of_car >= 0

    def get_gap(self, gap_ahead_of_car: np.ndarray) -> int:
        """
        return gap(nr of cells) between current and following vehicle
        """
        return np.where(gap_ahead_of_car)[0][0]

    def apply(self, state: np.ndarray) -> np.ndarray:
        for i, lane in enumerate(state):
            for index, speed in enumerate(lane):
                if speed > 0:
                    gap_ahead_of_car = self.check_following_vehicles(lane, index, speed)

                    if gap_ahead_of_car.any():
                        reduced_speed = self.get_gap(gap_ahead_of_car)
                        state[i, index] = reduced_speed

        return state


class Dawdling(AbstractRule):
    """
    reduce vehicle speed by 1 with the
    probability pd (dawning factor), if not already stationary (0)
    """

    def __init__(self, dawning_fac: int, seed: int):
        self.dawning_fac = dawning_fac
        self.seed = seed

    def apply(self, state: np.ndarray) -> np.ndarray:
        rand_gen = np.random.RandomState(self.seed)
        selected = rand_gen.choice([0, 1], state.shape, p=[1 - self.dawning_fac, self.dawning_fac])

        # only reduce cells with vehicles and non stationary vehicles
        check_speed = (state <= 0)
        selected[check_speed] = 0

        return state - selected


class MoveForward(AbstractRule):
    """
    move forward according to current speed
    """

    def get_new_position(self, lane: np.ndarray, index: int, speed: int) -> int:
        """
        calculate new position for vehicle
        """
        new_position = index + speed
        lane_len = lane.shape[0]
        if new_position >= lane_len:
            new_position -= lane_len

        return new_position

    def apply(self, state: np.ndarray) -> np.ndarray:
        # create empty street
        new_state = np.full_like(state, -1, dtype=int)

        for i, lane in enumerate(state):
            for index, speed in enumerate(lane):
                # calculate new position only if there is a vehicle
                if speed >= 0:
                    new_position = self.get_new_position(lane, index, speed)

                    # insert vehicle at updated position
                    new_state[i, new_position] = speed

        return new_state


class MergeBack(AbstractRule):
    def apply(self, state: np.ndarray) -> np.ndarray:
        return state
