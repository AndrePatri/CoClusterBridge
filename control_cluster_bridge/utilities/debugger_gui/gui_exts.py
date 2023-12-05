from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.defs import Journal
from control_cluster_bridge.utilities.debugger_gui.plot_utils import GridFrameWidget

from abc import abstractmethod

from typing import TypeVar

class SharedDataWindow():

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            grid_n_rows: int = -1,
            grid_n_cols: int = -1,
            window_buffer_factor: int = 2,
            namespace = "",
            name = "SharedDataWindow",
            parent: QWidget = None, 
            verbose = False):

        self.grid_n_rows = grid_n_rows
        self.grid_n_cols = grid_n_cols

        self.namespace = namespace

        self.name = name

        self.parent = parent

        self.update_data_dt = update_data_dt
        self.update_plot_dt = update_plot_dt
        self.window_duration = window_duration
        self.window_buffer_factor = window_buffer_factor

        self.journal = Journal()

        self.verbose = verbose

        self._reset()
        
    @abstractmethod
    def _initialize(self):
        
        pass

    @abstractmethod
    def _post_shared_init(self):
        
        # things to be done between shared data init and before ui initialization

        pass
    
    @abstractmethod
    def _init_shared_data(self):

        pass
    
    @abstractmethod
    def update(self):

        # this would tipically update each plotter in self.rt_plotters
        # using data from shared memory

        pass 
    
    def run(self):
        
        self._reset()

        self._init_shared_data()
        
        self._post_shared_init()

        self._init_ui()

    def _reset(self):

        self._terminated = False

        self.cluster_idx = 0

        self.shared_data_clients = []
        
        self.rt_plotters = []
        
    def _init_ui(self):

        self.grid = GridFrameWidget(self.grid_n_rows, self.grid_n_cols, 
            parent=self.parent)
        
        self.base_frame = self.grid.base_frame

        self.rt_plotters = []

        self._initialize()

    def swith_pause(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.paused = \
                    not self.rt_plotters[i].rt_plot_widget.paused

    def change_sample_update_dt(self, 
                dt: float):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.update_data_sample_dt(dt)

                self.rt_plotters[i].settings_widget.synch_max_window_size()

    def change_plot_update_dt(self, 
                    dt: float):
        
        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.set_timer_interval(dt)
                
    def nightshift(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.nightshift()
        
    def dayshift(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.dayshift()

    def terminate(self):

        for i in range(0, len(self.shared_data_clients)):

            self.shared_data_clients[i].terminate()
        
        self._terminated = True

SharedDataWindowChild = TypeVar('SharedDataWindowChild', bound='SharedDataWindow')
