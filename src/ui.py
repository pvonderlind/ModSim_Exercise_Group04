import numpy as np
import holoviews as hv
import panel as pn
from typing import Any
import colorcet as cc
from bokeh.models.formatters import PrintfTickFormatter
import param

from src.ca import Street, Runner
from src.rules import *

hv.extension('bokeh')
pn.extension()

class CurrentRunner(param.Parameterized):
    value = param.Parameter(None)

def plot_state(
        runner: Runner,
        timestep: int
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
    
    if runner is None:
        return pn.pane.Markdown('### No simulation history to display.')
    
    history = runner.history
    meters_per_cell = 4
    velocity_max = runner._street._v_max
    lanes = runner._street._lanes
    lane_len = runner._street._lane_len

    if timestep >= len(history) or timestep < 0:
        state = np.zeros((lanes, lane_len)) - 1
    else:
        state = history[timestep]

    gridded_data = {
        'Lane': np.arange(state.shape[0], dtype=int),
        'Meter': (np.arange(state.shape[1], dtype=int) * meters_per_cell),
        'Speed': state
    }
    
    color_map = cc.bgy
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

def create_simulation_parameter_info_card(
        runner: Runner
) -> pn.Card:
    """
    Creates a Card containing information about the runner's simulation parameters.
    
    Parameters
    ----------
    runner : Runner
        The runner to get the simulation parameters from.
        
    Returns
    -------
    pn.Card
        A Card containing information about the runner's simulation parameters.
    """
    
    if runner is None:
        return pn.pane.Markdown('')
    
    street = runner._street
    rules = runner._rule_list
    
    info = f'''
## Parameters

### Street
- Lanes: {street._lanes}
- Lane length: {street._lane_len} cells
- Cars: {street._n_cars}

### Rules'''
    
    for rule in rules:
        info += f'\n- {rule.__class__.__name__}: {rule.__dict__}'
    
    return pn.pane.Markdown(info, extensions=['nl2br'])


class TrafficSimulationUI:

    def __init__(self):
        self.ui = pn.template.MaterialTemplate(title='Traffic Simulation')
        
        # Sidebar contents
        self.ui.sidebar.append(pn.pane.Markdown('# Settings'))
        
        # Street parameter input widgets
        self.lane_len = pn.widgets.IntSlider(
            name='Lane length',
            start=100,
            value=250,
            end=1000,
            format=PrintfTickFormatter(format='%d cells'))
        
        self.lanes = pn.widgets.IntSlider(
            name='Lanes',
            start=1,
            value=1,
            end=10)
        
        self.n_cars = pn.widgets.IntSlider(
            name='Cars',
            start=1,
            value=20,
            end=100)
        
        self.ui.sidebar.append(
            pn.WidgetBox(
                '## Street',
                self.lane_len,
                self.lanes,
                self.n_cars))
        
        
        
        # Rules and rule parameter input widgets
        # Accelerate rule
        self.accelerate_checkbox = pn.widgets.Checkbox(name='Accelerate', value=True)
        self.v_max = pn.widgets.IntSlider(
            name='Maximum velocity',
            start=1,
            value=8,
            end=20,
            format=PrintfTickFormatter(format='%d cells'))
        
        # AvoidCollision rule
        self.avoid_collision_checkbox = pn.widgets.Checkbox(name='Avoid collision', value=True)
        
        # Dawdling rule
        self.dawdling_checkbox = pn.widgets.Checkbox(name='Dawdling', value=True)
        self.dawdling_factor = pn.widgets.FloatSlider(
            name='Dawdling factor',
            start=0.0,
            value=0.2,
            end=1.0,
            step=0.05,
            format=PrintfTickFormatter(format='%0.2f'))
        
        # MoveForward rule
        self.move_forward_checkbox = pn.widgets.Checkbox(name='Move forward', value=True)
        
        # MergeBack rule
        self.merge_back_checkbox = pn.widgets.Checkbox(name='Merge back', value=True)
        
        spacer_height = 15
        self.ui.sidebar.append(
            pn.WidgetBox(
                '## Rules',
                self.accelerate_checkbox,
                self.v_max,
                pn.Spacer(height=spacer_height),
                self.avoid_collision_checkbox,
                pn.Spacer(height=spacer_height),
                self.dawdling_checkbox,
                self.dawdling_factor,
                pn.Spacer(height=spacer_height),
                self.move_forward_checkbox,
                pn.Spacer(height=spacer_height),
                self.merge_back_checkbox))

        self.simulation_length = pn.widgets.IntSlider(
            name='Simulation length',
            start=1,
            value=250,
            end=1000,
            format=PrintfTickFormatter(format='%d timesteps'))
        
        self.run_simulation_button = pn.widgets.Button(name='Simulate', button_type='primary')
        self.simulation_progressb_bar = pn.widgets.Tqdm(sizing_mode='stretch_width', width_policy='max')
        self.run_simulation_button.on_click(self.run_simulation)
        
        self.ui.sidebar.extend([
            pn.Spacer(height=spacer_height),
            self.simulation_length,
            self.run_simulation_button,
            self.simulation_progressb_bar])
        
        # Main contents

        self.current_runner = CurrentRunner(value=None) #Runner(Street(self.lanes.value, self.lane_len.value, self.n_cars.value, self.v_max.value), [])
        
        self.timestep_player = pn.widgets.DiscretePlayer(
            name='Simulation History Player',
            options=[0],
            value=0,
            loop_policy='loop',
            align='center')

        timestep_label = pn.bind(
            lambda t: pn.pane.Markdown(
                f'- Timestep: {t}'),
            t=self.timestep_player)

        street_plot = pn.bind(
            plot_state,
            runner=self.current_runner.param.value,
            timestep=self.timestep_player)
        
        self.run_parameter_info = pn.bind(
            create_simulation_parameter_info_card,
            runner=self.current_runner.param.value)

        self.ui.main.extend([
            pn.Column(
                '## Simulation history',
                timestep_label,
                pn.Row(
                    street_plot,
                    align='center',
                    sizing_mode='stretch_width'),
                self.timestep_player,
                self.run_parameter_info,
                sizing_mode='stretch_width')])
        
    def get_user_interface(self):
        return self.ui

    def run_simulation(self, event: Any) -> None:
        """
        Runs the simulation.
        """
        
        street = Street(self.lanes.value, self.lane_len.value, self.n_cars.value, self.v_max.value)
        rules = []
        if self.accelerate_checkbox.value:
            rules.append(Accelerate(self.v_max.value))
        if self.avoid_collision_checkbox.value:
            rules.append(AvoidCollision())
        if self.dawdling_checkbox.value:
            rules.append(Dawdling(self.dawdling_factor.value))
        if self.move_forward_checkbox.value:
            rules.append(MoveForward())
        if self.merge_back_checkbox.value:
            rules.append(MergeBack())
        
        runner = Runner(street, rules, self.simulation_length.value)
        
        runner.run(tqdm_widget=self.simulation_progressb_bar)

        self.current_runner.value = runner
        
        self.timestep_player.options = list(range(len(runner.history)))
        self.timestep_player.value = self.timestep_player.options[0]

        print('Simulation complete.')
