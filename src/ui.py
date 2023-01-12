import numpy as np
import holoviews as hv
from holoviews.streams import Pipe
import panel as pn
from typing import Any
import colorcet as cc
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.models import FixedTicker
import param
import io
from datetime import datetime

from src.ca import Street, Runner
from src.rules import *

hv.extension('bokeh')
pn.extension(notifications=True)

METERS_PER_CELL = 4

class CurrentRunner(param.Parameterized):
    value = param.Parameter(None)

def gridded_data_from(state: np.ndarray, empty=False) -> dict:
    return {
            'y': np.arange(state.shape[0], dtype=int), # lanes
            'x': (np.arange(state.shape[1], dtype=int) * METERS_PER_CELL), # cells
            'z': state if not empty else (np.zeros_like(state, dtype=int) - 1) # speed
        }

def prepare_street_plot(runner: Runner, gridded_data_pipe: Pipe) -> hv.DynamicMap:
    
    # parameters defining the street
    lanes = runner._street._lanes
    lane_len = runner._street._lane_len
    velocity_max = runner._street._v_max
    
    # parameters defining the color mapping of car speeds
    color_map = cc.bgy
    custom_colormap = ['#cccccc'] + [color_map[i] for i in np.linspace(0, (len(color_map) - 1), (velocity_max + 1), dtype=int)]
    color_levels = (np.arange(-1, (velocity_max + 2)) - 0.5).tolist()
    colorbar_ticks = np.arange(-1, (velocity_max + 1)).tolist()

    heatmap_dmap = hv.DynamicMap(
        hv.HeatMap,
        streams=[gridded_data_pipe]
    )
    
    heatmap_dmap = heatmap_dmap.redim(
        x=hv.Dimension('Meter', range=(0, lane_len * METERS_PER_CELL), soft_range=(0, lane_len * METERS_PER_CELL)),
        y=hv.Dimension('Lane', range=((-0.5), (lanes - 0.5)), soft_range=(-0.5, (lanes - 0.5))),
        z=hv.Dimension('Speed', range=(-1, velocity_max), soft_range=(-1, velocity_max))
    )
    
    heatmap_dmap = heatmap_dmap
    
    
    last_10_percent_idx = int(lane_len * 0.9) * METERS_PER_CELL
    last_10_percent_line = hv.VLine(last_10_percent_idx)
    last_10_percent_line.opts(color='black', line_width=1)
    text = hv.Text(last_10_percent_idx, -0.4, ' throughput measurement')
    text.opts(text_font_size='8pt', text_color='black', text_align='left')
    
    street_plot = heatmap_dmap * last_10_percent_line * text
    street_plot.opts(
        hv.opts.Curve(default_tools=[]),
        hv.opts.HeatMap(
            responsive=True,
            aspect=4,
            max_height=400,
            default_tools=['reset'],
            #toolbar=None,
            cmap=custom_colormap,
            color_levels=color_levels,
            colorbar=True,
            colorbar_opts={ 'ticker': FixedTicker(ticks=colorbar_ticks) },
        )
    )
    
    return street_plot

def prepare_metric_plot(runner: Runner, timestep_player) -> hv.DynamicMap:
    speed_plot = hv.Curve(
        runner.metric_avg_rel_speed(),
        kdims=['Timestep'],
        vdims=[hv.Dimension('Avgerage relative speed', range=(-0.1, 1.1), soft_range=(-0.1, 1.1))]
    )
    speed_plot.opts(
        responsive=True,
        aspect=3,
        default_tools=['reset'],
        #toolbar=None
    )
    
    throughput_plot = hv.Curve(
        runner.metric_car_throughput(),
        kdims=['Timestep'],
        vdims=['car throughput']
    )
    throughput_plot.opts(
        color='orange',
        responsive=True,
        interpolation='steps-mid',
        aspect=3,
        default_tools=['reset'],
        #toolbar=None
    )
    
    # for some reason just using hv.VLine doesn't work, so i need to use this wrapper
    def create_timestep_slider(x):
        return hv.VLine(x)
    
    timestep_slider = hv.DynamicMap(create_timestep_slider, kdims='x', streams={'x':timestep_player[0].param.value})
    timestep_slider.opts(color='red', line_width=1)
    
    
    plot = ((speed_plot * timestep_slider) + (throughput_plot * timestep_slider))
    plot.opts(
        hv.opts.Layout(title='Metrics'),
    )
    
    return plot

def create_simulation_parameter_info_card(
        runner: Runner
) -> pn.Card:
    """
    Creates a Card containing information about the runner's simulation parameters.
    """
    
    if runner is None:
        return pn.pane.Markdown('')
    
    street = runner._street
    rules = runner._rule_list

    params_text = '### Street\n'
    params_text += f'- Lanes: {street._lanes}\n- Lane length: {street._lane_len} cells\n- Cars: {street._n_cars}'
    
    rules_text = '### Rules'
    
    for rule in rules:
        rules_text += f'\n- {rule.__class__.__name__}: {rule.__dict__}'
        
    simulation_text = f'\n### Simulation\n- Random seed: {runner._street._seed}\n- Timesteps: {len(runner.history)}'
    
    return pn.Card(
        pn.Row(
            pn.pane.Markdown(params_text, sizing_mode='stretch_width'), 
            pn.pane.Markdown(rules_text, sizing_mode='stretch_width'),
            pn.pane.Markdown(simulation_text, sizing_mode='stretch_width'),
            sizing_mode='stretch_width'),
        title='Parameters', sizing_mode='stretch_width')


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
        self.avoid_collision_checkbox = pn.widgets.Checkbox(name='Avoid collision/ Take Over', value=True)
        
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
        
        self.random_seed = pn.widgets.IntInput(
            name='Random seed',
            start=0,
            value=42,
            end=99999)
        
        self.run_simulation_button = pn.widgets.Button(name='Simulate', button_type='primary')
        self.simulation_progressb_bar = pn.widgets.Tqdm(text='Progress', sizing_mode='stretch_width', width_policy='max')
        self.run_simulation_button.on_click(self.run_simulation)
        
        self.ui.sidebar.append(
            pn.WidgetBox(
                '## Simulation',
                self.random_seed,
                self.simulation_length,
                pn.Spacer(height=spacer_height),
                self.run_simulation_button,
                self.simulation_progressb_bar))
        
        # Main contents
        
        initial_runner = Runner(
                Street(
                    self.lanes.value,
                    self.lane_len.value,
                    self.n_cars.value,
                    self.v_max.value,
                    self.random_seed.value),
                []
            )
        self.current_runner = CurrentRunner(value=initial_runner)
        self.gridded_data_pipe = Pipe(data=gridded_data_from(initial_runner._street._state, empty=True))
        
        self.export_simulation_button = pn.widgets.FileDownload(
            None,
            align='center')
        self.export_simulation_button.disabled = True
        self.export_simulation_button.aspect_ratio = 12
        
        self.timestep_player = pn.widgets.DiscretePlayer(
            name='Simulation History Player',
            options=[-1],
            value=-1,
            loop_policy='loop',
            align='center')

        street_plot = pn.bind(
            prepare_street_plot,
            runner=self.current_runner.param.value,
            gridded_data_pipe=self.gridded_data_pipe)
        
        metrics_plot = pn.bind(
            prepare_metric_plot,
            runner=self.current_runner.param.value,
            timestep_player=[self.timestep_player])
        
        self.run_parameter_info = pn.bind(
            create_simulation_parameter_info_card,
            runner=self.current_runner.param.value)
        
        @pn.depends(timestep=self.timestep_player, watch=True)
        def on_timestep_player_change(timestep):
            self.gridded_data_pipe.send(gridded_data_from(self.current_runner.value.history[timestep]))

        self.ui.main.extend([
            pn.Column(
                '## Simulation history',
                self.run_parameter_info,
                pn.Column(
                    street_plot,
                    metrics_plot,
                    align='center',
                    sizing_mode='stretch_width'),
                self.timestep_player,
                self.export_simulation_button,
                sizing_mode='stretch_width')])
        
    def get_user_interface(self):
        return self.ui

    def run_simulation(self, event: Any) -> None:
        """
        Runs the simulation.
        """
        try:
            
            street = Street(self.lanes.value, self.lane_len.value, self.n_cars.value, self.v_max.value, self.random_seed.value)
            rules = []
            if self.accelerate_checkbox.value:
                rules.append(Accelerate(self.v_max.value))
            if self.avoid_collision_checkbox.value:
                rules.append(BreakOrTakeOver())
            if self.dawdling_checkbox.value:
                rules.append(Dawdling(self.dawdling_factor.value, self.random_seed.value))
            if self.move_forward_checkbox.value:
                rules.append(MoveForward())
            if self.merge_back_checkbox.value:
                rules.append(MergeBack())
            
            runner = Runner(street, rules, self.simulation_length.value)
            
            runner.run(tqdm_widget=self.simulation_progressb_bar)

        except Exception as e:
            pn.state.notifications.error(f'Simulation failed: {e}. Details in console.', duration=10000)
            raise e
            
        self.current_runner.value = runner
        
        file = io.BytesIO()
        file.write(runner.serialize())
        file.seek(0)
        
        self.export_simulation_button.filename = f'traffic_jam_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin'
        self.export_simulation_button.file = file
        self.export_simulation_button.disabled = False
        self.export_simulation_button.button_type = 'primary'
        
        self.timestep_player.options = list(range(len(runner.history)))
        self.timestep_player.value = self.timestep_player.options[0]
