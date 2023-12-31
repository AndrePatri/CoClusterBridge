# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QSlider
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QFrame
from PyQt5.QtWidgets import QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap

from control_cluster_bridge.utilities.debugger_gui.plot_utils import RhcTaskRefWindow, RhcCmdsWindow, RhcStateWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import WidgetUtils
from control_cluster_bridge.utilities.shared_mem import SharedMemClient, SharedMemSrvr, SharedStringArray
from control_cluster_bridge.utilities.defs import launch_controllers_flagname
from control_cluster_bridge.utilities.defs import launch_keybrd_cmds_flagname
from control_cluster_bridge.utilities.defs import cluster_size_name, n_contacts_name
from control_cluster_bridge.utilities.defs import jnt_names_client_name, jnt_number_client_name
from control_cluster_bridge.utilities.defs import additional_data_name
from control_cluster_bridge.utilities.defs import env_selector_name
from control_cluster_bridge.utilities.sysutils import PathsGetter
from control_cluster_bridge.utilities.defs import Journal

import os

import torch

import sys

import time

from perf_sleep.pyperfsleep import PerfSleep

from typing import Callable

class SharedDataThread(QThread):

    trigger_update = pyqtSignal()
    samples_data_dt = pyqtSignal(float)

    def __init__(self, 
                update_dt: float, 
                verbose = True):
        
        super().__init__()

        self.journal = Journal()

        self.perf_timer = PerfSleep()

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
            
            # self.perf_timer.clock_sleep(int(actual_sleep * 1e9)) # from a child thread doesn't work

            time.sleep(actual_sleep)
        
    def run(self):

        if self.initialized:
            
            while not self._terminate:
                
                self._trigger_update()

class RtClusterDebugger(QMainWindow):

    def __init__(self, 
                namespace = "",
                data_update_dt: float = 0.01,
                plot_update_dt: float = 0.5, 
                window_length: float = 10.0, # [s]
                window_buffer_factor: int = 2,
                verbose: bool = False):

        self.journal = Journal()

        self.namespace = namespace
        
        self.widget_utils = WidgetUtils()

        self.data_update_dt = data_update_dt
        self.plot_update_dt = plot_update_dt

        self._tabs_terminated = [True] * 3

        self._terminated = False

        self._controllers_triggered = False
        self._keyboard_cmds_triggered = False

        self.window_length = window_length 
        self.window_buffer_factor = window_buffer_factor

        self.verbose = verbose

        self.app = QApplication(sys.argv)
    
        self.dark_mode_enabled = False

        self._paused = False

        # shared mem
        self.cluster_size_clnt = None
        self.n_contacts_clnt = None
        self.jnt_number_clnt = None
        self.jnt_names_clnt = None
        self.add_data_length_clnt = None
        self.launch_controllers = None
        self.launch_keyboard_cmds = None
        self.env_index = None

        # shared data
        self.shared_data_tabs_name = ["RhcTaskRefs", "RhcCmdRef", "RhcState"]
        self.shared_data_window = [None] * len(self.shared_data_tabs_name)
        self.shared_data_tabs_map = {}
        for i in range(len(self.shared_data_tabs_name)):
            self.shared_data_tabs_map[self.shared_data_tabs_name[i]] = \
                None

        super().__init__()

        self._init_add_shared_data()

        self._init_ui()     

        self._init_data_thread()

        self.show()
    
    def __del__(self):
        
        if not self._terminated:

            self.terminate()

    def terminate(self):

        # terminating additinal shared memory data
        if self.data_thread is not None:
            self.data_thread.terminate()

        if self.cluster_size_clnt is not None:
            self.cluster_size_clnt.terminate()

        if self.n_contacts_clnt is not None:
            self.n_contacts_clnt.terminate()

        if self.jnt_number_clnt is not None:
            self.jnt_number_clnt.terminate()
        
        if self.jnt_names_clnt is not None:
            self.jnt_names_clnt.terminate()
        
        if self.add_data_length_clnt is not None:
            self.add_data_length_clnt.terminate()
        
        if self.launch_controllers is not None:
            self.launch_controllers.terminate()
                
        if self.launch_keyboard_cmds is not None:
            self.launch_keyboard_cmds.terminate()

        if self.env_index is not None:
            self.env_index.terminate()

        # terminate shared data windows
        for i in range(len(self.shared_data_tabs_name)):

            if self.shared_data_window[i] is not None:

                self.shared_data_window[i].terminate()

        self._terminated = True

    def run(self):

        self.app.exec_()

    def _init_ui(self):

        self.setGeometry(100, 100, 1000, 800)
        self.setWindowTitle("Cluster real-time debugger")
        self.setContentsMargins(0, 0, 0, 0)

        self.central_widget = QWidget(self)
        self.central_widget.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter.setHandleWidth(1.0)
        self.layout.addWidget(self.splitter)
        
        self.tabs = self.widget_utils.ClosableTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs_layout = QVBoxLayout(self.tabs)
        # self.splitter_layout.addWidget(self.tabs)
        self.tabs.add_closing_method(self._terminate_tab)
        
        self.settings_frame = QFrame()
        self.settings_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.settings_frame.setFrameShape(QFrame.StyledPanel)
        self.settings_frame_layout = QVBoxLayout(self.settings_frame)  # Use QVBoxLayout here
        self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)
        # self.splitter_layout.addWidget(self.settings_frame)

        self.settings_label = QLabel("Settings")
        self.settings_frame_layout.addWidget(self.settings_label, 
                                    alignment=Qt.AlignCenter)

        self.cluster_idx_slider = self.widget_utils.generate_complex_slider(
                        parent=self.settings_frame, 
                        parent_layout=self.settings_frame_layout,
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{self.cluster_size - 1}", 
                        max=self.cluster_size - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="cluster index", 
                        callback=self._update_cluster_idx)
        
        paths = PathsGetter()
        icon_basepath = paths.GUI_ICONS_PATH

        self.trigger_controllers_button = self.widget_utils.create_iconed_button(
                            parent=self.settings_frame, 
                            parent_layout=self.settings_frame_layout,
                            icon_basepath=icon_basepath, 
                            icon="stop_controllers", 
                            icon_triggered="launch_controllers",
                            callback=self._toggle_controllers, 
                            descr="launch/stop controllers", 
                            size_x = 80, 
                            size_y = 50)
        
        self.nightshift_button = self.widget_utils.create_iconed_button(
                            parent=self.settings_frame, 
                            parent_layout=self.settings_frame_layout,
                            icon_basepath=icon_basepath, 
                            icon="nightshift", 
                            icon_triggered="dayshift",
                            callback=self._toggle_dark_mode, 
                            descr="dark/day mode")
        
        self.pause_button = self.widget_utils.create_iconed_button(
                            parent=self.settings_frame, 
                            parent_layout=self.settings_frame_layout,
                            icon_basepath=icon_basepath, 
                            icon="pause", 
                            icon_triggered="unpause",
                            callback=self._pause_all, 
                            descr="freeze/unfreeze plots")
        
        self.plot_update_dt_slider = self.widget_utils.generate_complex_slider(
                        parent=self.settings_frame, 
                        parent_layout=self.settings_frame_layout,
                        min_shown=f"{self.data_update_dt}", # sec.
                        min= int(self.data_update_dt * 1e3), # millisec. 
                        max_shown=f"{1.0}", 
                        max=int(1.0 * 1e3), 
                        init_val_shown=f"{self.plot_update_dt}", 
                        init=int(self.plot_update_dt * 1e3), 
                        title="plot update dt [s]", 
                        callback=self._change_plot_update_dt)

        self.samples_update_dt_slider = self.widget_utils.generate_complex_slider(
                        parent=self.settings_frame, 
                        parent_layout=self.settings_frame_layout,
                        min_shown=f"{0.001}", # sec.
                        min= int(0.001 * 1e3), # millisec. 
                        max_shown=f"{1.0}", 
                        max=int(1.0 * 1e3), 
                        init_val_shown=f"{self.data_update_dt}", 
                        init=int(self.data_update_dt * 1e3), 
                        title="samples update dt [s]", 
                        callback=self._change_samples_update_dt)
        
        self.trigger_keyboard_cmds_button = self.widget_utils.create_iconed_button(
                            parent=self.settings_frame, 
                            parent_layout=self.settings_frame_layout,
                            icon_basepath=icon_basepath, 
                            icon="stop_keyboard_cmds", 
                            icon_triggered="start_keyboard_cmds",
                            callback=self._toggle_keyboard_cmds, 
                            descr="launch/stop cmds from keyboard", 
                            size_x = 80, 
                            size_y = 50)
        
        self.data_spawner = self.widget_utils.create_scrollable_list(parent=self.settings_frame, 
                                                parent_layout=self.settings_frame_layout, 
                                                list_names=self.shared_data_tabs_name, 
                                                callback=self._spawn_shared_data_tabs, 
                                                default_checked=False, 
                                                title="data spawner")

        self.settings_frame_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.splitter.addWidget(self.tabs)

        self.splitter.addWidget(self.settings_frame)

    def _init_data_thread(self):

        self.data_thread = SharedDataThread(self.data_update_dt)
        
        self.data_thread.trigger_update.connect(self._update_from_shared_data,
                                        Qt.QueuedConnection)
        
        self.data_thread.start()

    def _init_add_shared_data(self):

        wait_amount = 0.05
        
        # getting info
        self.cluster_size_clnt = SharedMemClient(name=cluster_size_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.cluster_size_clnt.attach()
        self.n_contacts_clnt = SharedMemClient(name=n_contacts_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.n_contacts_clnt.attach()
        self.jnt_number_clnt = SharedMemClient(name=jnt_number_client_name(), 
                                        namespace=self.namespace, 
                                        dtype=torch.int64, 
                                        wait_amount=wait_amount, 
                                        verbose=self.verbose)
        self.jnt_number_clnt.attach()
        self.jnt_names_clnt = SharedStringArray(length=self.jnt_number_clnt.tensor_view[0, 0].item(), 
                                    name=jnt_names_client_name(), 
                                    namespace=self.namespace, 
                                    is_server=False, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.jnt_names_clnt.start()
        self.add_data_length_clnt = SharedMemClient(name=additional_data_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self.verbose)
        self.add_data_length_clnt.attach()

        self.launch_controllers = SharedMemClient(name=launch_controllers_flagname(), 
                                namespace=self.namespace, 
                                dtype=torch.bool, 
                                client_index=0,
                                verbose=self.verbose)

        self.launch_controllers.attach()
        self.launch_controllers.set_bool(False) # by default don't trigger the controllers

        self.env_index = SharedMemSrvr(n_rows=1, n_cols=1, 
                                name=env_selector_name(), 
                                namespace=self.namespace, 
                                dtype=torch.int64)
        self.env_index.start()
        self.env_index.tensor_view[0, 0] = 0 # inizialize to 1st environment

        self.cluster_size = self.cluster_size_clnt.tensor_view[0, 0].item()
        self.n_contacts = self.n_contacts_clnt.tensor_view[0, 0].item()
        self.jnt_names = self.jnt_names_clnt.read()
        self.jnt_number = self.jnt_number_clnt.tensor_view[0, 0].item()
        self.add_data_length = self.add_data_length_clnt.tensor_view[0, 0].item()
    
    def _spawn_shared_data_tabs(self, 
                    label: str):
        
        if label == self.shared_data_tabs_name[0]:
                
                checked = self.data_spawner.buttons[0].isChecked()  # Get the new state of the button

                if checked and self._tabs_terminated[0]:
                    
                    self.shared_data_window[0] = RhcTaskRefWindow(update_data_dt=self.data_update_dt, 
                                        update_plot_dt=self.plot_update_dt,
                                        window_duration=self.window_length, 
                                        cluster_size=self.cluster_size,
                                        n_contacts=self.n_contacts,
                                        window_buffer_factor=self.window_buffer_factor, 
                                        namespace=self.namespace,
                                        parent=None, 
                                        verbose = self.verbose)

                    self.tab_rhc_task_refs = QWidget()
                    self.tab_rhc_task_refs_layout = QVBoxLayout()
                    self.tab_rhc_task_refs.setLayout(self.tab_rhc_task_refs_layout)

                    self.shared_data_tabs_map[self.shared_data_tabs_name[0]] = self.tabs.count()
                    self.tabs.addTab(self.tab_rhc_task_refs, 
                        "RhcTaskRef")
                    
                    self.tab_rhc_task_refs_layout.addWidget(self.shared_data_window[0].base_frame)

                    # self.data_spawner.buttons[0].setStyleSheet("")  # Reset button style

                    self._tabs_terminated[0] = False

                    self.data_spawner.buttons[0].setCheckable(False)

                    self.data_spawner.buttons[0].setChecked(False)
                    
                    self._update_dark_mode()

                # if not checked:

                #     self.data_spawner.buttons[0].setStyleSheet("color: darkgray")  # Change button style to dark gray
        
        if label == self.shared_data_tabs_name[1]:
                
                checked = self.data_spawner.buttons[1].isChecked()  # Get the new state of the button

                if checked and self._tabs_terminated[1]:
                    
                    self.shared_data_window[1] = RhcCmdsWindow(update_data_dt=self.data_update_dt, 
                                    update_plot_dt=self.plot_update_dt,
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    add_data_length=self.add_data_length,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    namespace=self.namespace,
                                    parent=None, 
                                    verbose = self.verbose)
                    
                    self.tab_rhc_cmds = QWidget()
                    self.tab_rhc_cmds_layout = QVBoxLayout()
                    self.tab_rhc_cmds.setLayout(self.tab_rhc_cmds_layout)

                    self.shared_data_tabs_map[self.shared_data_tabs_name[1]] = self.tabs.count()
                    self.tabs.addTab(self.tab_rhc_cmds, 
                        "RhcCmds")

                    self.tab_rhc_cmds_layout.addWidget(self.shared_data_window[1].base_frame)

                    # self.data_spawner.buttons[1].setStyleSheet("")  # Reset button style

                    self._tabs_terminated[1] = False

                    self.data_spawner.buttons[1].setCheckable(False)

                    self.data_spawner.buttons[1].setChecked(False)

                    self._update_dark_mode()

                # if not checked:

                #     self.data_spawner.buttons[1].setStyleSheet("color: darkgray")  # Change button style to dark gray
        
        if label == self.shared_data_tabs_name[2]:
                
                checked = self.data_spawner.buttons[2].isChecked()  # Get the new state of the button

                if checked and self._tabs_terminated[2]:
                    
                    self.shared_data_window[2] = RhcStateWindow(update_data_dt=self.data_update_dt, 
                                    update_plot_dt=self.plot_update_dt,
                                    window_duration=self.window_length, 
                                    cluster_size=self.cluster_size,
                                    jnt_names=self.jnt_names, 
                                    jnt_number=self.jnt_number,
                                    window_buffer_factor=self.window_buffer_factor, 
                                    namespace=self.namespace,
                                    parent=None, 
                                    verbose = self.verbose)

                    self.tab_rhc_state = QWidget()
                    self.tab_rhc_state_layout = QVBoxLayout()
                    self.tab_rhc_state.setLayout(self.tab_rhc_state_layout)

                    self.shared_data_tabs_map[self.shared_data_tabs_name[2]] = self.tabs.count()
                    self.tabs.addTab(self.tab_rhc_state, 
                                    "RhcState")
                    
                    self.tab_rhc_state_layout.addWidget(self.shared_data_window[2].base_frame)

                    # self.data_spawner.buttons[2].setStyleSheet("")  # Reset button style

                    self._tabs_terminated[2] = False
                    
                    self.data_spawner.buttons[2].setCheckable(False)

                    self.data_spawner.buttons[2].setChecked(False)

                    self._update_dark_mode()

                # if not checked:

                    # self.data_spawner.buttons[2].setStyleSheet("color: darkgray")  # Change button style to dark gray

    def _pause_all(self):
        
        for i in range(len(self.shared_data_tabs_name)):

            if not self._tabs_terminated[i]:

                self.shared_data_window[i].swith_pause()

        self._paused = not self._paused

        if self._paused:

            self.pause_button.iconed_button.setIcon(self.pause_button.triggered_icone_button)

        if not self._paused:

            self.pause_button.iconed_button.setIcon(self.pause_button.icone_button)

    def _change_plot_update_dt(self, 
                    millisec: int):
        
        dt_sec = float(millisec * 1e-3)

        for i in range(0, len(self.shared_data_tabs_name)):

            self.shared_data_window[i].change_plot_update_dt(dt_sec)
       
        self.plot_update_dt_slider.current_val.setText(f'{dt_sec:.3f}')

    def _change_samples_update_dt(self, 
                    millisec: int):

        dt_sec = float(millisec * 1e-3)

        self.data_thread.samples_data_dt.emit(dt_sec)

        self.samples_update_dt_slider.current_val.setText(f'{dt_sec:.4f}')
        
        for i in range(len(self.shared_data_tabs_name)):

            self.shared_data_window[i].change_sample_update_dt(dt_sec)
        
    def _update_from_shared_data(self):
        
        for i in range(len(self.shared_data_tabs_name)):

            if not self._tabs_terminated[i] and \
                self.shared_data_window[i] is not None:

                self.shared_data_window[i].update()

    def _update_cluster_idx(self, 
                        idx: int):

        self.cluster_idx_slider.current_val.setText(f'{idx}')

        self.env_index.tensor_view[0, 0] = idx

        for i in range(len(self.shared_data_tabs_name)):

            if not self._tabs_terminated[i]:

                self.shared_data_window[i].cluster_idx = idx

    def _toggle_controllers(self):
        
        self._controllers_triggered = not self._controllers_triggered
            
        self.launch_controllers.set_bool(self._controllers_triggered)

        if self._controllers_triggered:
            
            self.trigger_controllers_button.iconed_button.setIcon(self.trigger_controllers_button.triggered_icone_button)

        if not self._controllers_triggered:

            self.trigger_controllers_button.iconed_button.setIcon(self.trigger_controllers_button.icone_button)
    
    def _connect_to_keyboard_cmds(self):

        self.launch_keyboard_cmds = SharedMemClient(name=launch_keybrd_cmds_flagname(), 
                        namespace=self.namespace, 
                        dtype=torch.bool, 
                        client_index=0, 
                        verbose=self.verbose)
        self.launch_keyboard_cmds.attach()
        self.launch_keyboard_cmds.set_bool(False) # we disable cmds by default

    def _toggle_keyboard_cmds(self):

        self._keyboard_cmds_triggered = not self._keyboard_cmds_triggered
        
        if self.launch_keyboard_cmds is None:
            
            # we spawn the necessary shared mem client if not already done
            self._connect_to_keyboard_cmds()

        self.launch_keyboard_cmds.set_bool(self._keyboard_cmds_triggered)

        if self._keyboard_cmds_triggered:
            
            self.trigger_keyboard_cmds_button.iconed_button.setIcon(
                self.trigger_keyboard_cmds_button.triggered_icone_button)

        if not self._keyboard_cmds_triggered:

            self.trigger_keyboard_cmds_button.iconed_button.setIcon(
                self.trigger_keyboard_cmds_button.icone_button)

    def _update_dark_mode(self):

        stylesheet = self._get_stylesheet()

        self.setStyleSheet(stylesheet)

        if self.dark_mode_enabled:
            
            self.nightshift_button.iconed_button.setIcon(self.nightshift_button.triggered_icone_button)

            for i in range(len(self.shared_data_tabs_name)):

                if self.shared_data_window[i] is not None:

                    self.shared_data_window[i].nightshift()

        if not self.dark_mode_enabled:
            
            self.nightshift_button.iconed_button.setIcon(self.nightshift_button.icone_button)

            for i in range(len(self.shared_data_tabs_name)):

                if self.shared_data_window[i] is not None:

                    self.shared_data_window[i].dayshift()

    def _toggle_dark_mode(self):

        self.dark_mode_enabled = not self.dark_mode_enabled

        self._update_dark_mode()

    def _get_stylesheet(self):

        text_color = "#ffffff" if self.dark_mode_enabled else "#333333"
        background_color = "#333333" if self.dark_mode_enabled else "#FFFFFF"
        
        return f"""
            QFrame {{
                background-color: {background_color};
                padding: 0px;
            }}
            QLabel {{
                color: {text_color};
                background-color: {background_color};
                padding: 0px;
            }}
            QPushButton {{
                background-color: {background_color}
                color: {text_color};
                padding: 0px;
            }}
            QPushButton::icon {{
                background-color: {background_color};
            }}
            QSlider {{
                background-color: {background_color};
                color: {text_color};
            }}
            QIcon {{
                background-color: {background_color};
                color: {text_color};
                padding: 0px;
            }}
            QTabWidget::pane {{
                background-color: {background_color};
                color: {text_color};
                padding: 0px;
            }}
            QTabWidget::tab-bar {{
                background-color: {background_color};
                color: {text_color};
                padding: 0px;
            }}
            QTabBar::tab {{
                background-color: {background_color};
                color: {text_color};
                padding: 0px;
            }}
            QTabBar::tab:selected {{
                background-color: {background_color};
                padding: 0px;
            }}
            QScrollBar:vertical {{
                background: {background_color};
                color: {text_color};
            }}
            QScrollBar:horizontal {{
                background: {background_color};
                color: {text_color};
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background: {background_color};
                color: {text_color};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                background: {background_color};
                color: {text_color};
            }}
            QScrollArea {{
                background: {background_color};
                color: {text_color};
            }}

        """
    
    def _get_key_by_value(self, 
                dictionary, target_value):

        for key, value in dictionary.items():

            if value == target_value:

                return key
            
        return None 

    def closeEvent(self, event):
        # This function is called when the window is being closed
        # You can perform your desired actions here

        message = f"[{self.__class__.__name__}]" + f"[{self.journal.status}]: " \
                + f"closing debugger and performing some cleanup..."
            
        print(message)

        self.terminate()

        event.accept()  

    def _terminate_tab(self, 
                    tab_idx: int):
        
        print(self.shared_data_tabs_map)
        index = self.shared_data_tabs_name.index(
                            self._get_key_by_value(self.shared_data_tabs_map, \
                                        tab_idx))

        self._tabs_terminated[index] = True

        for i in range(len(self.shared_data_window)):

            if index == i and self.shared_data_window[i] is not None:

                self.shared_data_window[i].terminate()

                self.data_spawner.buttons[i].setChecked(False)

                self.data_spawner.buttons[i].setCheckable(True)

if __name__ == "__main__":  

    data_update_dt = 0.01
    plot_update_dt = 0.1

    window_length = 10.0
    window_buffer_factor = 2

    main_window = RtClusterDebugger(data_update_dt=data_update_dt,
                            plot_update_dt=plot_update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True)

    main_window.run()

