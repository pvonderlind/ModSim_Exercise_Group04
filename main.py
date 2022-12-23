from src.ca import *
from src.rules import DummyShuffleRule

if __name__ == "__main__":
    street = Street(1, 250, 20, 8)
    rules = [DummyShuffleRule()]
    runner = Runner(street, rules)
    runner.run()
    print(len(runner.history))