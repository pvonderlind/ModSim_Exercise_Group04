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


class BreakOrTakeOver(AbstractRule):
    """
    reduce vehicle speed to gap size, 
    if gap to next vehicle is smaller than its speed

    If there is space to the left of the vehicle with the un-braked
    speed, it is allowed to take over and switch to the left lane.
    """

    def apply(self, state: np.ndarray) -> np.ndarray:
        cars_indices_that_took_over = []
        for i, lane in enumerate(state):
            for index, speed in enumerate(lane):
                if index in cars_indices_that_took_over:
                    continue
                if speed > 0:
                    gap_ahead_of_car = self.check_following_vehicles(lane, index, speed)

                    if gap_ahead_of_car.any():
                        # Car is ahead, check to take over from the left, leftmost lane can't take over
                        if (i < (state.shape[0] - 1)) and self.is_left_lane_clear(state, i, index, speed):
                            state[i, index] = -1
                            state[i + 1, index] = speed
                            cars_indices_that_took_over.append(index)
                        else:
                            reduced_speed = self.get_gap(gap_ahead_of_car)
                            state[i, index] = reduced_speed

            cars_indices_that_took_over = []
        return state

    def check_following_vehicles(self, lane: np.ndarray, index: int, speed: int) -> np.ndarray:
        """
        returns array of which following cells contain a vehicle
        """
        # move current vehicle to front
        shifted_state = np.roll(lane, -(index + 1))
        gap_ahead_of_car = shifted_state[:speed]
        return gap_ahead_of_car >= 0

    def is_left_lane_clear(self, state: np.ndarray, lane_idx: int, index: int, speed: int) -> bool:
        left_gap = self.check_following_vehicles(state[lane_idx + 1], index, speed)
        return not left_gap.any()

    def get_gap(self, gap_ahead_of_car: np.ndarray) -> int:
        """
        return gap(nr of cells) between current and following vehicle
        """
        return np.where(gap_ahead_of_car)[0][0]


class Dawdling(AbstractRule):
    """
    reduce vehicle speed by 1 with the
    probability pd (dawning factor), if not already stationary (0)
    """

    def __init__(self, dawning_fac: int, seed: int):
        self.dawning_fac = dawning_fac
        self.rand_gen = np.random.RandomState(seed)

    def apply(self, state: np.ndarray) -> np.ndarray:
        selected = self.rand_gen.choice([0, 1], state.shape, p=[1 - self.dawning_fac, self.dawning_fac])

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

        # NOTE: 0 is our rightmost lane in the visualization!
        for i, lane in enumerate(state):
            # Last lane can't swap places since we do the swapping in reverse order
            # This is necessary to not allow multiple switches for one car in one call of this rule.
            if i == state.shape[0] - 1:
                break
            for index, speed in enumerate(lane):
                if state[i, index] == -1:
                    if state[i + 1, index] != -1:
                        state[i, index] = state[i + 1, index]
                        state[i + 1, index] = -1
        return state
