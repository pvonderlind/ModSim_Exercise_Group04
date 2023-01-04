from src.ca import *
from src.rules import DummyShuffleRule, Accelerate, AvoidCollision

if __name__ == "__main__":
    v_max = 8
    street = Street(1, 250, 20, v_max)
    rules = [DummyShuffleRule(), Accelerate(v_max), AvoidCollision()]
    runner = Runner(street, rules)
    runner.run()
    print(len(runner.history))