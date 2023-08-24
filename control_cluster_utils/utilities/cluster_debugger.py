from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QSlider
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QFrame
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QScrollArea, QPushButton, QScrollBar, QSpacerItem, QSizePolicy, QSlider

from control_cluster_utils.utilities.plot_utils import RhcTaskRefWindow, RhcCmdsWindow, RhcStateWindow
from control_cluster_utils.utilities.shared_mem import SharedMemClient, SharedStringArray
from control_cluster_utils.utilities.defs import cluster_size_name, n_contacts_name
from control_cluster_utils.utilities.defs import jnt_names_client_name, jnt_number_client_name
from control_cluster_utils.utilities.defs import additional_data_name
from control_cluster_utils.utilities.sysutils import PathsGetter
from PyQt5.QtGui import QIcon, QPixmap

import os

import torch

import sys

import time

from typing import Callable

class SharedDataThread(QThread):

    trigger_update = pyqtSignal()
    samples_data_dt = pyqtSignal(float)

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
        
        self.samples_data_dt.connect(self._update_sampling_dt,
                                        Qt.QueuedConnection)
        
        self.initialized = True
    
    def terminate(self):

        self._terminate = True

    def _update_sampling_dt(self, 
                        dt: float):
        
        self.update_dt = dt

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
                data_update_dt: float = 0.01,
                plot_update_dt: float = 0.5, 
                window_length: float = 10.0, # [s]
                window_buffer_factor: int = 2,
                verbose: bool = False):

        self.data_update_dt = data_update_dt
        self.plot_update_dt = plot_update_dt

        self._tabs_terminated = [False] * 3

        self._terminated = False

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
    
    def closeEvent(self, event):
        # This function is called when the window is being closed
        # You can perform your desired actions here

        message = f"[{self.__class__.__name__}]" + f"[status]: " \
                + f"closing debugger and performing some cleanup..."
            
        print(message)

        self.terminate()

        event.accept()  

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
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter.setHandleWidth(0.1)
        self.layout.addWidget(self.splitter)
        
        self.tabs = ClosableTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        self.settings_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.settings_frame.setFrameShape(QFrame.StyledPanel)
        self.settings_frame_layout = QVBoxLayout(self.settings_frame)  # Use QVBoxLayout here
        self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)
        # self.splitter_layout.addWidget(self.settings_frame)

        self.settings_label = QLabel("Settings")
        self.settings_frame_layout.addWidget(self.settings_label, 
                                    alignment=Qt.AlignCenter)

        self._generate_slider(parent=self.settings_frame, 
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{self.rhc_cmds_plotter.cluster_size - 1}", 
                        max=self.rhc_cmds_plotter.cluster_size - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="cluster index", 
                        callback=self.update_cluster_idx)
        
        paths = PathsGetter()
        icon_basepath = paths.GUI_ICONS_PATH
        self._create_iconed_button(parent=self.settings_frame, 
                            icon_basepath=icon_basepath, 
                            icon="pause", 
                            callback=self._pause_all, 
                            descr="freeze/unfreeze all")
        
        self._generate_slider(parent=self.settings_frame, 
                        min_shown=f"{self.data_update_dt}", # sec.
                        min= int(self.data_update_dt * 1e3), # millisec. 
                        max_shown=f"{1.0}", 
                        max=int(1.0 * 1e3), 
                        init_val_shown=f"{self.plot_update_dt}", 
                        init=int(self.plot_update_dt * 1e3), 
                        title="plot update dt [s]", 
                        callback=self.change_plot_update_dt)

        self._generate_slider(parent=self.settings_frame, 
                        min_shown=f"{0.001}", # sec.
                        min= int(0.001 * 1e3), # millisec. 
                        max_shown=f"{1.0}", 
                        max=int(1.0 * 1e3), 
                        init_val_shown=f"{self.data_update_dt}", 
                        init=int(self.data_update_dt * 1e3), 
                        title="samples update dt [s]", 
                        callback=self.change_samples_update_dt)
        
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

    def _create_iconed_button(self, 
                    parent: QWidget, 
                    icon_basepath: str, 
                    icon: str,
                    callback: Callable[[int], None], 
                    descr: str = ""):
        iconpath = os.path.join(icon_basepath, 
                                   icon + ".svg")
        
        button_frame = QFrame(parent)
        button_frame.setFrameShape(QFrame.StyledPanel)
        button_frame_color = button_frame.palette().color(parent.backgroundRole()).name()
        button_layout = QHBoxLayout(button_frame)  # Use QVBoxLayout here
        button_layout.setContentsMargins(2, 2, 2, 2)

        button_descr = QLabel(descr)
        button_descr.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        button = QPushButton(button_frame)
        button.setGeometry(100, 100, 100, 50)
        button.setStyleSheet(f"background-color: {button_frame_color};")
        pixmap = QPixmap(iconpath)

        button_icon = QIcon(pixmap)

        button.setIcon(button_icon)
        button.setFixedSize(30, 30)
        button.setIconSize(button.size())
        
        button.clicked.connect(callback)
    
        button_layout.addWidget(button_descr)
        button_layout.addWidget(button)
        self.settings_frame_layout.addWidget(button_frame)
    
    def _init_windows(self):

        self.rhc_task_plotter = RhcTaskRefWindow(update_data_dt=self.data_update_dt, 
                                    update_plot_dt=self.plot_update_dt,
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    n_contacts=self.n_contacts,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_cmds_plotter = RhcCmdsWindow(update_data_dt=self.data_update_dt, 
                                    update_plot_dt=self.plot_update_dt,
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    add_data_length=self.add_data_length,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)
        
        self.rhc_states_plotter = RhcStateWindow(update_data_dt=self.data_update_dt, 
                                    update_plot_dt=self.plot_update_dt,
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    parent=None, 
                                    verbose = self.verbose)

    def _init_data_thread(self):

        self.data_thread = SharedDataThread(self.data_update_dt)
        
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
    
    def _pause_all(self):

        if not self._tabs_terminated[0]:

            self.rhc_task_plotter.pause()

        if not self._tabs_terminated[1]:

            self.rhc_cmds_plotter.pause()

        if not self._tabs_terminated[2]:

            self.rhc_states_plotter.pause()
    
    def change_plot_update_dt(self, 
                    millisec: int):
        
        dt_sec = float(millisec * 1e-3)
        self.rhc_task_plotter.change_plot_update_dt(dt_sec)
        self.rhc_cmds_plotter.change_plot_update_dt(dt_sec)
        self.rhc_states_plotter.change_plot_update_dt(dt_sec)

        self.sliders_current_vals[1].setText(f'{dt_sec:.3f}')

    def change_samples_update_dt(self, 
                    millisec: int):

        dt_sec = float(millisec * 1e-3)

        self.data_thread.samples_data_dt.emit(dt_sec)

        self.sliders_current_vals[2].setText(f'{dt_sec:.4f}')
        
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
        
        if not self._terminated:

            self.terminate()

    def terminate(self):

        self.data_thread.terminate()

        self.cluster_size_clnt.terminate()
        self.n_contacts_clnt.terminate()
        self.jnt_number_clnt.terminate()
        self.jnt_names_clnt.terminate()
        self.add_data_length_clnt.terminate()

        self.rhc_task_plotter.terminate()
        self.rhc_cmds_plotter.terminate()
        self.rhc_states_plotter.terminate()

        self._terminated = True

if __name__ == "__main__":  

    update_dt = 0.05
    window_length = 4.0
    window_buffer_factor = 1
    main_window = RtClusterDebugger(update_dt=update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True)

    main_window.run()
