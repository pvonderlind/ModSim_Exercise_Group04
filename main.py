import panel as pn
from src.ui import TrafficSimulationUI

PORT = 8001

if __name__ == "__main__":
    sim = TrafficSimulationUI()
    pn.serve(sim.get_user_interface, port=PORT)
