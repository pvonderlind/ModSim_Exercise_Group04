from src.ca import *
from src.rules import DummyShuffleRule, Accelerate, AvoidCollision, Dawdling, MoveForward

if __name__ == "__main__":
    v_max = 8
    dawning_fac = 0.2
    street = Street(1, 250, 20, v_max)
    rules = [Accelerate(v_max), AvoidCollision(), Dawdling(dawning_fac), MoveForward()]
    runner = Runner(street, rules)
    runner.run()
    print(len(runner.history))