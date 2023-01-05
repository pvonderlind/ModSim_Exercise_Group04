import panel as pn
from src.ui import TrafficSimulationUI

if __name__ == "__main__":
    sim = TrafficSimulationUI()
    pn.serve(sim.ui)