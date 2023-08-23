from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QSlider
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QFrame
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy

from control_cluster_utils.utilities.plot_utils import RhcTaskRefWindow, RhcCmdsWindow, RhcStateWindow
from control_cluster_utils.utilities.shared_mem import SharedMemClient, SharedStringArray
from control_cluster_utils.utilities.defs import cluster_size_name, n_contacts_name
from control_cluster_utils.utilities.defs import jnt_names_client_name, jnt_number_client_name
from control_cluster_utils.utilities.defs import additional_data_name

import torch

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
        
        self.sliders_current_vals = []
        
        super().__init__()

        self._init_shared_data()

        self._init_windows()

        self._init_ui()     

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
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(0.1)
        self.layout.addWidget(self.splitter)
        
        self.tabs = ClosableTabWidget()
        # self.splitter_layout.addWidget(self.tabs)
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
        
        self.settings_frame = QFrame()
        self.settings_frame.setFrameShape(QFrame.StyledPanel)
        self.settings_frame_layout = QVBoxLayout(self.settings_frame)  # Use QVBoxLayout here
        self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)
        # self.splitter_layout.addWidget(self.settings_frame)
        self.settings_lavel = QLabel("Settings")

        self._generate_slider(parent=self.settings_frame, 
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{self.rhc_cmds_plotter.cluster_size - 1}", 
                        max=self.rhc_cmds_plotter.cluster_size - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="cluster index", 
                        callback=self.update_cluster_idx)
        
        self.settings_frame_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self.settings_frame)

    def _generate_slider(self, 
                    parent: QWidget, 
                    callback: Callable[[int], None], 
                    min_shown: str, 
                    min: int,
                    max_shown: str, 
                    max: int,
                    init_val_shown: str,
                    init: int,
                    title: str):

            val_frame = QFrame(parent)
            val_frame.setFrameShape(QFrame.StyledPanel)
            val_layout = QHBoxLayout(val_frame)  # Use QVBoxLayout here
            val_layout.setContentsMargins(2, 2, 2, 2)

            val_title = QLabel(title)
            current_val = QLabel(init_val_shown)
            current_val.setAlignment(Qt.AlignRight)
            current_val.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")

            val_layout.addWidget(val_title, 
                                    alignment=Qt.AlignLeft)
            val_layout.addWidget(current_val)

            val_slider_frame = QFrame(parent)
            val_slider_frame.setFrameShape(QFrame.StyledPanel)
            val_slider_frame_layout = QHBoxLayout(val_slider_frame)  # Use QHBoxLayout here
            val_slider_frame_layout.setContentsMargins(2, 2, 2, 2)

            min_label = QLabel(min_shown)
            min_label.setAlignment(Qt.AlignCenter)
            min_label.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")
            val_slider_frame_layout.addWidget(min_label)

            val_slider = QSlider(Qt.Horizontal)
            val_slider.setMinimum(min)
            val_slider.setMaximum(max)
            val_slider.setValue(init)
            val_slider.valueChanged.connect(callback)
            val_slider_frame_layout.addWidget(val_slider)

            max_label = QLabel(max_shown)
            max_label.setAlignment(Qt.AlignCenter)
            max_label.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")
            val_slider_frame_layout.addWidget(max_label)

            self.settings_frame_layout.addWidget(val_frame)
            self.settings_frame_layout.addWidget(val_slider_frame)
            
            self.tab_rhc_task_refs_layout.addWidget(self.rhc_task_plotter.base_frame)
            self.tab_rhc_cmds_layout.addWidget(self.rhc_cmds_plotter.base_frame)
            self.tab_rhc_state_layout.addWidget(self.rhc_states_plotter.base_frame)

            self.sliders_current_vals.append(current_val)

    def _init_windows(self):

        self.rhc_task_plotter = RhcTaskRefWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    n_contacts=self.n_contacts,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_cmds_plotter = RhcCmdsWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    add_data_length=self.add_data_length,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_states_plotter = RhcStateWindow(update_dt=self.update_dt, 
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)

    def _init_data_thread(self):

        self.data_thread = SharedDataThread(self.update_dt)
        
        self.data_thread.trigger_update.connect(self.update_from_shared_data,
                                        Qt.QueuedConnection)
        self.data_thread.start()

    def _init_shared_data(self):

        wait_amount = 0.05
        
        # getting info
        self.cluster_size_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.cluster_size_clnt.attach()
        self.n_contacts_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=n_contacts_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=True)
        self.n_contacts_clnt.attach()
        self.jnt_number_clnt = SharedMemClient(n_rows=1, n_cols=1,
                                        name=jnt_number_client_name(), 
                                        dtype=torch.int64, 
                                        wait_amount=wait_amount, 
                                        verbose=self.verbose)
        self.jnt_number_clnt.attach()
        self.jnt_names_clnt = SharedStringArray(length=self.jnt_number_clnt.tensor_view[0, 0].item(), 
                                    name=jnt_names_client_name(), 
                                    is_server=False, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.jnt_names_clnt.start()
        self.add_data_length_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=additional_data_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=True)
        self.add_data_length_clnt.attach()

        self.cluster_size = self.cluster_size_clnt.tensor_view[0, 0].item()
        self.n_contacts = self.n_contacts_clnt.tensor_view[0, 0].item()
        self.jnt_names = self.jnt_names_clnt.read()
        self.jnt_number = self.jnt_number_clnt.tensor_view[0, 0].item()
        self.add_data_length = self.add_data_length_clnt.tensor_view[0, 0].item()
    
    def update_from_shared_data(self):
        
        if not self._tabs_terminated[0]:

            self.rhc_task_plotter.update()

        if not self._tabs_terminated[1]:

            self.rhc_cmds_plotter.update()

        if not self._tabs_terminated[2]:

            self.rhc_states_plotter.update()

    def update_cluster_idx(self, 
                        idx: int):

        self.sliders_current_vals[0].setText(f'{idx}')

        if not self._tabs_terminated[0]:

            self.rhc_task_plotter.cluster_idx = idx
        
        if not self._tabs_terminated[1]:

            self.rhc_cmds_plotter.cluster_idx = idx

        if not self._tabs_terminated[2]:
            
            self.rhc_states_plotter.cluster_idx = idx

    def run(self):

        self.app.exec_()

    def __del__(self):

        self.terminate()

    def terminate(self):

        self.data_thread.terminate()

        self.cluster_size_clnt.terminate()
        self.n_contacts_clnt.terminate()
        self.jnt_number_clnt.terminate()
        self.jnt_names_clnt.terminate()
        self.add_data_length_clnt.terminate()

if __name__ == "__main__":  

    update_dt = 0.05
    window_length = 4.0
    window_buffer_factor = 1
    main_window = RtClusterDebugger(update_dt=update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True)

    main_window.run()
