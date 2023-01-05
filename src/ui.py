import numpy as np
import holoviews as hv
import panel as pn
from typing import Any
import colorcet as cc

from ca import Street, Runner
from rules import *

hv.extension('bokeh')
pn.extension()


def plot_state(
        history: np.ndarray,
        timestep: int,
        color_map: Any = cc.bgy,
        meters_per_cell=4,
        velocity_max=6,
        lanes=1,
        lane_len=100
) -> hv.HeatMap:
    """
    Plots the Cellular Automata history at a certain point in time as a HeatMap.
    
    Parameters
    ----------
    state : np.ndarray
        The state to plot. Shape is (lanes, cells).
    color_map : List[str], optional
        A list of colors to use for the HeatMap, by default Colorcet's BGY color map.
    meters_per_cell : int, optional
        The number of meters per cell, by default 4.
    velocity_max : int, optional
        The maximum velocity, by default 6.
        
    Returns
    -------
    hv.HeatMap
        A plot representing the state of the Cellular Automata via a HeatMap.
    """

    if timestep >= len(history) or timestep < 0:
        state = np.zeros((lanes, lane_len)) - 1
    else:
        state = history[timestep]

    gridded_data = {
        'Lane': np.arange(state.shape[0], dtype=int),
        'Meter': (np.arange(state.shape[1], dtype=int) * meters_per_cell),
        'Speed': state
    }

    custom_colormap = ['#CCC'] + [color_map[i] for i in np.linspace(0, len(color_map) - 1, velocity_max, dtype=int)]

    plot = hv.HeatMap(
        gridded_data,
        kdims=['Meter', 'Lane'],
        vdims=hv.Dimension('Speed', range=(-1, velocity_max), soft_range=(-1, velocity_max))
    )

    plot.opts(
        responsive=True,
        aspect=min(4, state.shape[1]) / state.shape[0],
        max_height=400,
        default_tools=[],
        # tools=['hover'],
        toolbar=None,
        # xticks=gridded_data['Meter'],
        yticks=gridded_data['Lane'],
        cmap=custom_colormap,
        color_levels=velocity_max + 1,
        colorbar=True
    )

    return plot


class TrafficSimulationUI:

    def ui(self):
        ui = pn.template.MaterialTemplate(title='Traffic Simulation')

        run_simulation_button = pn.widgets.Button(name='Simulate', button_type='primary')

        self.timestep_player = pn.widgets.DiscretePlayer(
            name='Simulation History Player',
            options=[0],
            value=0,
            loop_policy='loop',
            align='center')

        lanes = 1
        lane_len = 100
        n_cars = 20
        v_max = 8
        dawning_fac = 0.2

        street = Street(1, 250, 20, v_max)
        rules = [Accelerate(v_max), AvoidCollision(), Dawdling(dawning_fac), MoveForward()]
        runner = Runner(street, rules)
        self.runner = Runner(street, rules)

        street_plot = pn.bind(
            plot_state,
            history=self.runner.history,
            timestep=self.timestep_player,
            velocity_max=v_max,
            lanes=lanes,
            lane_len=lane_len)

        timestep_label = pn.bind(
            lambda t: pn.pane.Markdown(
                f"Timestep: {t}"),
            t=self.timestep_player)

        ui.main.extend([
            pn.Column(
                '## Traffic Simulation',
                timestep_label,
                pn.Row(
                    street_plot,
                    align='center',
                    sizing_mode='stretch_width'),
                self.timestep_player,
                sizing_mode='stretch_width')])

        run_simulation_button.on_click(self.run_simulation)

        ui.sidebar.extend([
            pn.WidgetBox(
                '## Settings',
                'Click "Simulate" to compute the simulation. Then on the right, click the "Play" replay the simulation history.',
                run_simulation_button)])

        return ui

    def run_simulation(self, event: Any) -> None:
        """
        Runs the simulation.
        """

        self.runner.run()

        self.timestep_player.options = list(range(len(self.runner.history)))

        print('Simulation complete.')


if __name__ == "__main__":
    sim = TrafficSimulationUI()
    pn.serve(sim.ui)
