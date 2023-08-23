from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout

from control_cluster_utils.utilities.plot_utils import RhcTaskRefWindow, RhcCmdsWindow, RhcStateWindow

import sys

import time

from typing import Callable

class SharedDataThread(QThread):

    trigger_update = pyqtSignal()

    def __init__(self, 
                update_dt: float, 
                verbose = True):
        
        super().__init__()

        self._cluster_index = 0 # data for this cluster will be emitted

        self.update_dt = update_dt

        self._terminate = False
        self.initialized = False
        
        self.verbose = verbose
        
        self.rhc_task_refs = []
        self.rhc_cmd = []
        self.rhc_state = []
                
        self.initialized = True

    def terminate(self):

        self._terminate = True

    def _trigger_update(self):
        
        t = time.perf_counter()

        self.trigger_update.emit()
        
        update_duration = time.perf_counter() - t # compensate for emit time

        actual_sleep = self.update_dt - update_duration

        if (actual_sleep > 0):

            time.sleep(actual_sleep)
        
    def run(self):

        if self.initialized:
            
            while not self._terminate:
                
                self._trigger_update()

    def update_cluster_idx(self, 
                    idx: int):

        if idx < self.cluster_size:

            self._cluster_index = idx

class ClosableTabWidget(QTabWidget):

    def __init__(self):

        super().__init__()

        self.setTabsClosable(True)

        self.tabCloseRequested.connect(self.close_tab)

    def close_tab(self, 
            currentIndex):

        if currentIndex != -1:

            widget = self.widget(currentIndex)

            if widget is not None:

                widget.deleteLater()

            self.removeTab(currentIndex)

            self.perform_other_closing_steps(currentIndex)

    def add_closing_method(self, 
            callback: Callable[[int], None]):

        self.perform_other_closing_steps = callback

class RtClusterDebugger(QMainWindow):

    def __init__(self, 
                update_dt: float = 0.1, 
                window_length: float = 10.0, # [s]
                window_buffer_factor: int = 2,
                verbose: bool = False):

        self._tabs_terminated = [False] * 3

        self.update_dt = update_dt
        self.window_length = window_length 
        self.window_buffer_factor = window_buffer_factor

        self.verbose = verbose

        self.app = QApplication(sys.argv)
        
        super().__init__()

        self._init_ui()
        
        self._init_windows()

        self._init_data_thread()

        self.show()
    
    def terminate_tab(self, 
                    tab_idx: int):
        
        self._tabs_terminated[tab_idx] = True

        if tab_idx == 0:

            self.rhc_task_plotter.terminate()

        if tab_idx == 1:
            
            self.rhc_cmds_plotter.terminate()

        if tab_idx == 2:

            self.rhc_states_plotter.terminate()            

    def _init_ui(self):

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Cluster real-time debugger")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)
        
        self.tabs = ClosableTabWidget()
        self.layout.addWidget(self.tabs)
        self.tabs.add_closing_method(self.terminate_tab)

        self.tab_rhc_task_refs = QWidget()
        self.tab_rhc_cmds = QWidget()
        self.tab_rhc_state = QWidget()
        self.tab_rhc_task_refs_layout = QVBoxLayout()
        self.tab_rhc_cmds_layout = QVBoxLayout()
        self.tab_rhc_state_layout = QVBoxLayout()
        self.tab_rhc_task_refs.setLayout(self.tab_rhc_task_refs_layout)
        self.tab_rhc_cmds.setLayout(self.tab_rhc_cmds_layout)
        self.tab_rhc_state.setLayout(self.tab_rhc_state_layout)

        self.tabs.addTab(self.tab_rhc_task_refs, 
                        "RhcTaskRef")
        self.tabs.addTab(self.tab_rhc_cmds, 
                        "RhcCmds")
        self.tabs.addTab(self.tab_rhc_state, 
                        "RhcState")
        
    def _init_windows(self):

        self.rhc_task_plotter = RhcTaskRefWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_cmds_plotter = RhcCmdsWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_states_plotter = RhcStateWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.tab_rhc_task_refs_layout.addWidget(self.rhc_task_plotter.base_frame)
        self.tab_rhc_cmds_layout.addWidget(self.rhc_cmds_plotter.base_frame)
        self.tab_rhc_state_layout.addWidget(self.rhc_states_plotter.base_frame)

    def _init_data_thread(self):

        self.data_thread = SharedDataThread(self.update_dt)
        
        self.data_thread.trigger_update.connect(self.update_from_shared_data,
                                        Qt.QueuedConnection)
        self.data_thread.start()

    def update_from_shared_data(self):
        
        if not self._tabs_terminated[0]:

            self.rhc_task_plotter.update()

        if not self._tabs_terminated[1]:

            self.rhc_cmds_plotter.update()

        if not self._tabs_terminated[2]:

            self.rhc_states_plotter.update()

    def update_cluster_idx(self, 
                        idx: int):

        self.rhc_task_plotter.cluster_idx = idx

    def run(self):

        self.app.exec_()

    def __del__(self):

        self.terminate()

    def terminate(self):

        self.data_thread.terminate()

if __name__ == "__main__":  

    update_dt = 0.05
    window_length = 4.0
    window_buffer_factor = 1
    main_window = RtClusterDebugger(update_dt=update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True)

    main_window.run()
