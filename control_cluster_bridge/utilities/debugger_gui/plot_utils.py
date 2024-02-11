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
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTabWidget
from PyQt5.QtWidgets import QHBoxLayout, QFrame
from PyQt5.QtWidgets import QScrollArea, QPushButton, QSpacerItem, QSizePolicy, QSlider
from PyQt5.QtWidgets import QSplitter, QLabel
from PyQt5.QtGui import QIcon, QPixmap

import pyqtgraph as pg

from typing import List, Callable, Union

from control_cluster_bridge.utilities.sysutils import PathsGetter

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

import os

class RtPlotWidget(pg.PlotWidget):

    def __init__(self, 
                window_duration: int,
                n_dims: int,  
                update_data_dt: float, 
                update_plot_dt: float, 
                base_name: str, 
                n_data: int = 1, # in case of matrix plot
                legend_list: List[str] = None,
                parent: QWidget = None, 
                xlabel = "sample n.", 
                ylabel = "", 
                window_buffer_factor: int = 2):

        super().__init__(title=base_name,
                    parent=parent)
        
        self.plot_item = self.getPlotItem()

        if legend_list is not None and len(legend_list) != n_dims:
            
            Journal.log(self.__class__.__name__,
                "__init__",
                f"provided legend list length {len(legend_list)} does not match data dimension {n_dims}",
                LogType.EXCEP,
                throw_when_excep = True)
            
        self.legend_list = legend_list

        self.x_label = xlabel
        self.y_label = ylabel

        self.paused = False

        self.nightmode = False

        # self.ntimestamps_per_window = 10
        
        self.n_dims = n_dims # number of dimensions of the single sample (the one visualized)

        self.n_data = n_data # number of "timelines" to be stored. Useful for plotting time-series
        # of matrices
        self.current_data_index = 0 # only one data can be plotted at a time

        self.window_offset = 0

        self.window_size = int(window_duration // update_data_dt) + 1 # floor division
        self.window_duration = window_duration

        self.window_buffer_factor = window_buffer_factor
        self.window_buffer_size = self.window_buffer_factor * self.window_size
        self.window_buffer_duration = self.window_buffer_factor * self.window_duration

        self.base_name = base_name

        self.update_data_dt = update_data_dt
        self.update_plot_dt = update_plot_dt

        self.data = np.zeros((self.window_buffer_size, self.n_dims, self.n_data))

        self.sample_stamps = np.arange(0, self.window_buffer_size)

        self.labels = [] 
        self.lines = []

        self._setup_plot()

        self._init_lines()

        self.update_window_size(self.window_size)

        # self.update_timestamps_res(self.ntimestamps_per_window)
        
        self._init_timers()
    
    def update(self, 
            new_data: np.ndarray,
            data_idx: int = 0
            ):

        # updates window with new data

        if not self.paused:

            self.data[:, :, :] = np.roll(self.data, -1, axis=0) # roll data "pages" backwards
            # for each page (first dimension) data is arranges in a matrix
            # [data dim x data sample]

            updated_data_shape = new_data.shape
            data_size = len(updated_data_shape)

            if data_size == 2:
                
                # data in assumed to be of shape data_dim x num_data (data_idx is not used)

                if updated_data_shape[0] != self.n_dims:
                    
                    exep = f"Provided data n. rows should be equal to {self.n_dims}, " + \
                        f"but got {updated_data_shape[0]}"
                    
                    raise Exception(exep)
                
                if updated_data_shape[1] != self.n_data:
                    
                    exep = f"Provided data n. cols should be equal to {self.n_data}, " + \
                        f"but got {updated_data_shape[1]}"
                    
                    raise Exception(exep)
                
                # update last sample
                self.data[-1, :, :] = new_data

            if data_size == 1:

                if updated_data_shape[0] != self.n_dims:
                    
                    exep = f"Provided data length should be equal to {self.n_dims}, " + \
                        f"but got {updated_data_shape[0]}"
                    
                    raise Exception(exep)

                # update last sample at provided data idx(if not default)
                self.data[-1, :, data_idx] = new_data
            
            if data_size == 0:

                exep = f"Cannot update time-series with 0-D data"
                
                raise Exception(exep)
            
            if data_size > 2:

                exep = f"Can only plot time-series of vectors (1D) or matrices (2D)"
                
                raise Exception(exep)

    def switch_to_data(self,
            data_idx: int):

        if data_idx > (self.n_data - 1):
            
            exep = f"Provided data index {data_idx} exceeds max. val. of {self.n_data - 1}"

            raise Exception(exep)
        
        self.current_data_index = data_idx

    def set_timer_interval(self, 
                    sec: float):

        self.timer.setInterval(int(sec * 1e3)) # millisec.
        self.timer.start()

    def hide_line(self, 
                index: int):

        self.lines[index].hide() 

    def show_line(self, 
                index: int):

        self.lines[index].show() 

    def update_data_sample_dt(self, 
                        dt: float):
        
        self.update_data_dt = dt
        
    def update_window_size(self, 
                new_size: int):
        
        self.window_size = min(new_size, self.window_buffer_size)

        x_range = (self.window_buffer_size - 1 - self.window_size - self.window_offset * self.window_size, 
            self.window_buffer_size - 1 - self.window_offset * self.window_size) 
        
        self.setXRange(*x_range)

    def update_window_offset(self, 
                    offset: int = 0):
        
        if offset > self.window_buffer_size - self.window_size:

            offset = self.window_buffer_size - self.window_size

        self.window_offset = offset

        x_range = (self.window_buffer_size - 1 - self.window_size - self.window_offset, 
            self.window_buffer_size - 1 - self.window_offset) 
        
        self.setXRange(*x_range)

    def update_timestamps_res(self, 
                    res: int = 10):
        
        self.ntimestamps_per_window = res
    
    def nightshift(self):

        self.setBackground('k')

        title_style = {'color': 'w', 'size': '10pt'}
        self.plotItem.setTitle(title=self.base_name, **title_style)

        tick_color = pg.mkColor('w')  # Use 'b' for blue color, you can replace it with your preferred color
        self.x_axis.setPen(tick_color)
        self.x_axis.setTextPen(tick_color)
        self.y_axis.setPen(tick_color)
        self.y_axis.setTextPen(tick_color)

    def dayshift(self):

        self.setBackground('w')

        title_style = {'color': 'k', 'size': '10pt'}
        self.plotItem.setTitle(title=self.base_name, **title_style)

        tick_color = pg.mkColor('k')  # Use 'b' for blue color, you can replace it with your preferred color
        self.x_axis.setPen(tick_color)
        self.x_axis.setTextPen(tick_color)
        self.y_axis.setPen(tick_color)
        self.y_axis.setTextPen(tick_color)

    def _init_lines(self):

        for i in range(0, self.n_dims):
            
            if self.legend_list is None:

                label = f"{self.base_name}_{i}"  # generate label for each line

            else:
                
                label = self.legend_list[i]

            self.labels.append(label)

            color = self.colors[i] 
            color.setAlpha(255)  # Set the alpha value for the color

            pen = pg.mkPen(color=color, 
                    width=2.3)

            self.lines.append(self.plot_item.plot(self.data[:, i, self.current_data_index], 
                        pen=pen))
        
    def _setup_plot(self):
        
        self.enableAutoRange()

        self.x_axis = self.plotItem.getAxis('bottom')
        self.y_axis = self.plotItem.getAxis('left')

        self.plotItem.setLabel('left', self.y_label)
        self.plotItem.setLabel('bottom', self.x_label)

        # self.setDownsampling(auto= True, mode = None, ds = None)

        # Set grid color to black
        self.showGrid(x=True, y=True, alpha=1.0)  # Full opacity for grid lines
        self.x_axis.setGrid(255)  
        self.y_axis.setGrid(255) 

        # Define a list of colors for each row
        self.colors = [pg.intColor(i, self.n_dims, 255) for i in range(self.n_dims)]

        self.dayshift() # sets uppearance for light mode
    
    def _contrasting_colors(self, 
                        num_colors: int):

        colors = []

        for i in range(num_colors):

            hue = (i / num_colors) * 360  # Spread hues across the color wheel

            color = pg.mkColor(hue, 255, 200)  # Use full saturation and lightness for contrast

            colors.append(color)

        return colors

    def _init_timers(self):
        
        self.timer = QTimer()
        self.timer.setInterval(int(self.update_plot_dt * 1e3)) # millisec.
        self.timer.timeout.connect(self._update_plot_data)
        self.timer.start()
    
    def _update_plot_data(self):
        
        for i in range(0, self.n_dims):

            self.lines[i].setData(self.data[:, i, self.current_data_index])

    def _update_timestams_ticks(self, 
                        elapsed_times: List[float]):
        
        # display only some of the x-axis timestamps to avoid overlap
        step_size = max(1, int(self.window_size // self.ntimestamps_per_window))

        selected_labels = elapsed_times[::step_size]

        x_tick_vals = [i for i, _ in enumerate(elapsed_times) if i % step_size == 0]
        x_tick_names = [f'{t:.2f}s' for t in selected_labels]

        x_ticks = []
        for i in range(0, len(selected_labels)):
            
            x_ticks.append((x_tick_vals[i], 
                            x_tick_names[i]))

        self.getAxis('bottom').setTicks([x_ticks])
    
class WidgetUtils:

    class SliderData:

        def __init__(self):
            
            self.base_frame = None
            self.base_layout = None
            self.val_frame = None
            self.val_layout = None
            self.val_slider_frame = None
            self.val_slider_frame_layout = None
            self.min_label = None
            self.max_label = None
            self.val_slider = None
            self.current_val = None
            self.title = None

    class IconedButtonData:

        def __init__(self):
            
            self.clicked_iconpath = None
            self.iconed_button_frame = None
            self.iconed_button_layout = None
            self.iconed_button = None
            self.icone_button = None
            self.triggered_icone_button = None
            self.pixmap = None
            self.button_descr = None
            self.iconpath = None
    
    class ScrollableListButtonData:

        def __init__(self):

            self.buttons = []

    class ScrollableListLabelsData:

        def __init__(self):
            
            self.base_frame = None
            self.base_layout = None
            self.labels = []
        
        def update(self, 
                data: np.ndarray):

            if data.ndim != 1:

                raise Exception("Provided data should be a 1D numpy array!")
            
            for i in range(0, len(data)):
                
                # for now using np.round
                self.labels[i].setText(str((round(data[i], 4))))

    def generate_complex_slider(self, 
                callback: Callable[[int], None],
                min_shown: str,
                min: int,
                max_shown: str, 
                max: int,
                init_val_shown: str,
                init: int,
                title: str,
                parent: QWidget = None,
                parent_layout: Union[QHBoxLayout, QVBoxLayout] = None):

        slider_data = self.SliderData()

        base_frame = QFrame(parent)
        base_frame.setFrameShape(QFrame.StyledPanel)
        base_frame.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        base_layout = QVBoxLayout(base_frame)  # Use QVBoxLayout here
        base_layout.setContentsMargins(0, 0, 0, 0)

        val_frame = QFrame(base_frame)
        val_frame.setFrameShape(QFrame.StyledPanel)
        val_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        val_layout = QHBoxLayout(val_frame)  # Use QVBoxLayout here
        val_layout.setContentsMargins(2, 2, 2, 2)

        val_title = QLabel(title)
        current_val = QLabel(init_val_shown)
        current_val.setAlignment(Qt.AlignRight)
        current_val.setStyleSheet("border: 1px solid gray; border-radius: 4px;")
        val_title.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        current_val.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        val_layout.addWidget(val_title, 
                                alignment=Qt.AlignLeft)
        val_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        val_layout.addWidget(current_val)

        val_slider_frame = QFrame(base_frame)
        val_slider_frame.setFrameShape(QFrame.StyledPanel)
        val_slider_frame_layout = QHBoxLayout(val_slider_frame)  # Use QHBoxLayout here
        val_slider_frame_layout.setContentsMargins(2, 2, 2, 2)
        val_slider_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        min_label = QLabel(min_shown)
        min_label.setAlignment(Qt.AlignCenter)
        min_label.setStyleSheet("border: 1px solid gray; border-radius: 4px;")
        val_slider_frame_layout.addWidget(min_label)
        min_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        val_slider = QSlider(Qt.Horizontal)
        val_slider.setMinimum(min)
        val_slider.setMaximum(max)
        val_slider.setValue(init)
        val_slider.valueChanged.connect(callback)
        val_slider_frame_layout.addWidget(val_slider)
        min_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        max_label = QLabel(max_shown)
        max_label.setAlignment(Qt.AlignCenter)
        max_label.setStyleSheet("border: 1px solid gray; border-radius: 4px;")
        val_slider_frame_layout.addWidget(max_label)
        max_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        base_layout.addWidget(val_frame)
        base_layout.addWidget(val_slider_frame)
        
        if parent_layout is not None:
            parent_layout.addWidget(base_frame)

        slider_data.base_frame = base_frame
        slider_data.base_layout = base_layout
        slider_data.val_frame = val_frame
        slider_data.val_layout = val_layout
        slider_data.val_slider_frame = val_slider_frame
        slider_data.val_slider_frame_layout = val_slider_frame_layout
        slider_data.min_label = min_label
        slider_data.max_label = max_label
        slider_data.val_slider = val_slider
        slider_data.current_val = current_val
        slider_data.title = title 

        return slider_data
    
    def create_iconed_button(self, 
                parent: QWidget, 
                parent_layout: Union[QHBoxLayout, QVBoxLayout],
                icon_basepath: str, 
                icon: str,
                callback: Callable[[int], None], 
                icon_triggered: str = None,
                descr: str = "", 
                size_x: int= 30, 
                size_y: int = 30):
        
        button_data = self.IconedButtonData()

        button_frame = QFrame(parent)
        button_frame.setFrameShape(QFrame.StyledPanel)
        button_frame.setGeometry(100, 100, 200, 200)
        button_layout = QHBoxLayout(button_frame)  # Use QVBoxLayout here
        button_layout.setContentsMargins(2, 2, 2, 2)

        button_descr = QLabel(descr)
        button_descr.setAlignment(Qt.AlignLeft | Qt.AlignCenter)

        button = QPushButton(button_frame)
        button.setGeometry(100, 100, 100, 50)

        iconpath = os.path.join(icon_basepath, 
                                   icon + ".svg")
        pixmap = QPixmap(iconpath)
        button_icon = QIcon(pixmap)

        if icon_triggered is not None:
            
            iconpath_triggered = os.path.join(icon_basepath, 
                                   icon_triggered + ".svg")
            pixmap_triggered = QPixmap(iconpath_triggered)
            triggered_button_icon = QIcon(pixmap_triggered)

        else:

            triggered_button_icon = None

        button.setIcon(button_icon)
        button.setFixedSize(size_x, size_y)
        button.setIconSize(button.size())
        
        button.clicked.connect(callback)
        
        button_data.iconpath = iconpath
        button_data.iconed_button_frame = button_frame
        button_data.iconed_button_layout = button_layout
        button_data.iconed_button = button
        button_data.icone_button = button_icon
        button_data.triggered_icone_button = triggered_button_icon
        button_data.pixmap = pixmap
        button_data.button_descr = button_descr

        button_layout.addWidget(button_descr)
        button_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button_layout.addWidget(button)
        
        parent_layout.addWidget(button_frame)

        return button_data
    
    def create_scrollable_list_button(self, parent: QWidget,
                                parent_layout: Union[QHBoxLayout, QVBoxLayout], 
                                list_names: List[str], 
                                callback: Callable[[str], None], 
                                toggle_all_callback: Callable[[], None] = None, 
                                title: str = "", default_checked=True):

        list_data = self.ScrollableListButtonData()

        # Create the main scroll area
        plot_selector_scroll_area = QScrollArea(parent=parent)
        plot_selector_scroll_area.setWidgetResizable(True)
        
        # Create the frame and layout for the list
        list_frame = QFrame(plot_selector_scroll_area)
        list_frame.setFrameShape(QFrame.StyledPanel)
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(2, 2, 2, 2)

        # Create a horizontal layout for the title and the show/hide all button
        title_layout = QHBoxLayout()
        plot_selector_title = QLabel(title, 
                                alignment=Qt.AlignHCenter)
        title_layout.addWidget(plot_selector_title)

        # Create the show/hide all button
        if toggle_all_callback is not None:

            show_hide_all_button = QPushButton("Show/hide all", list_frame)
            show_hide_all_button.clicked.connect(toggle_all_callback)
            title_layout.addWidget(show_hide_all_button)

        # Add the title layout to the list layout
        list_layout.addLayout(title_layout)

        # Add legend buttons to the list
        for label in list_names:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(default_checked)
            button.clicked.connect(lambda checked, l=label: callback(l))
            list_data.buttons.append(button)
            list_layout.addWidget(button)
        
        # Finalize the scroll area setup
        plot_selector_scroll_area.setWidget(list_frame)
        list_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        parent_layout.addWidget(plot_selector_scroll_area)

        return list_data

    def create_scrollable_label_list(self, 
                    parent: QWidget, 
                    parent_layout: Union[QHBoxLayout, QVBoxLayout],
                    list_names: List[str], 
                    title: str = "", 
                    init: List[float] = None
                    ):
        
        data = self.ScrollableListLabelsData()

        plot_selector_scroll_area = QScrollArea(parent=parent)
        plot_selector_scroll_area.setWidgetResizable(True)
        
        list_frame = QFrame(plot_selector_scroll_area)
        list_frame.setFrameShape(QFrame.StyledPanel)
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(2, 2, 2, 2)
        
        data.base_frame = list_frame
        data.base_layout = list_layout
        
        plot_selector_title = QLabel(title)
        list_layout.addWidget(plot_selector_title, 
                                alignment=Qt.AlignHCenter)

        # Add legend buttons to the plot selector frame
        i = 0

        for label in list_names:

            val_frame = QFrame(list_frame)
            val_frame.setFrameShape(QFrame.StyledPanel)
            val_layout = QHBoxLayout(val_frame)  # Use QVBoxLayout here
            val_layout.setContentsMargins(2, 2, 2, 2)

            val_title = QLabel(label)
            current_val = QLabel(str(init[i]))
            current_val.setAlignment(Qt.AlignRight)
            current_val.setStyleSheet("border: 1px solid gray; border-radius: 4px;")

            val_layout.addWidget(val_title, 
                                alignment=Qt.AlignLeft)
            val_layout.addWidget(current_val)

            data.labels.append(current_val)

            list_layout.addWidget(val_frame)
            
            i = i + 1

        plot_selector_scroll_area.setWidget(list_frame)
        
        list_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        if parent_layout is not None:

            parent_layout.addWidget(plot_selector_scroll_area)

        return data

    def generate_name_val_list(self, 
                parent: QWidget, 
                parent_layout: Union[QHBoxLayout, QVBoxLayout],
                callback: Callable[[int], None],
                title: List[str],
                init: List[float] = None):

        slider_data = self.NameValPairList()

        val_frame = QFrame(parent)
        val_frame.setFrameShape(QFrame.StyledPanel)
        val_layout = QHBoxLayout(val_frame)  # Use QVBoxLayout here
        val_layout.setContentsMargins(2, 2, 2, 2)

        val_title = QLabel(title)
        current_val = QLabel(str(init))
        current_val.setAlignment(Qt.AlignRight)
        current_val.setStyleSheet("border: 1px solid gray; border-radius: 4px;")

        val_layout.addWidget(val_title, 
                            alignment=Qt.AlignLeft)
        val_layout.addWidget(current_val)

        val_slider = QSlider(Qt.Horizontal)
        val_slider.setValue(init)
        val_slider.valueChanged.connect(callback)
        
        parent_layout.addWidget(val_frame)

        slider_data.val_frame = val_frame
        slider_data.val_layout = val_layout
        slider_data.val_slider = val_slider
        slider_data.current_val = current_val
        slider_data.title = title 

        return slider_data
    
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

class SettingsWidget():

    def __init__(self, 
            rt_plotter: RtPlotWidget,
            parent: QWidget = None
            ):

        self.rt_plot_widget = rt_plotter

        self.widget_utils = WidgetUtils()

        self.frame = QFrame(parent=parent)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setContentsMargins(0, 0, 0, 0)
        self.settings_frame_layout = QVBoxLayout(self.frame)  # Use QVBoxLayout here
        self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.window_size_slider = None
        self.window_offset_slider = None

        self.pause_button = None

        self.plot_selector = None

        self.all_lines_visible = False

        self.paused = False

        self._init_ui()

    def _init_ui(self):
        
        paths = PathsGetter()
        icon_basepath = paths.GUI_ICONS_PATH
        self.pause_button = self.widget_utils.create_iconed_button(parent=self.frame, 
                                parent_layout=self.settings_frame_layout,
                                icon_basepath=icon_basepath, 
                                icon="pause", 
                                icon_triggered="unpause",
                                descr="freeze/unfreeze", 
                                callback=self.change_pause_state)
        
        self.window_size_slider = self.widget_utils.generate_complex_slider(title="window size [s]: ", 
                                parent=self.frame, 
                                parent_layout=self.settings_frame_layout,
                                callback=self.update_window_size, 
                                min_shown=f'{self.rt_plot_widget.update_data_dt}', 
                                min = 1,
                                max_shown=f'{self.rt_plot_widget.window_buffer_duration}', 
                                max = self.rt_plot_widget.window_buffer_size,
                                init_val_shown =f'{self.rt_plot_widget.update_data_dt * self.rt_plot_widget.window_size}', 
                                init=self.rt_plot_widget.window_size)

        self.window_offset_slider = self.widget_utils.generate_complex_slider(title="window offset [n.samples]: ", 
                                parent=self.frame, 
                                parent_layout=self.settings_frame_layout,
                                callback=self.update_window_offset, 
                                min_shown=f'{0}', 
                                min = 0,
                                max_shown=f'{self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size}', 
                                max = self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size,
                                init_val_shown =f'{self.rt_plot_widget.window_offset}', 
                                init=self.rt_plot_widget.window_offset)

        self.plot_selector = self.widget_utils.create_scrollable_list_button(parent=self.frame, 
                                        parent_layout=self.settings_frame_layout,
                                        list_names=self.rt_plot_widget.labels, 
                                        callback=self.toggle_line_visibility, 
                                        toggle_all_callback=self.toggle_all_visibility,
                                        title="line selector")

        self.settings_frame_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def change_pause_state(self):
        
        self.paused = not self.paused
        
        if self.paused: 
            
            self.pause_button.iconed_button.setIcon(self.pause_button.triggered_icone_button)

        else:

            self.pause_button.iconed_button.setIcon(self.pause_button.icone_button)

        self.rt_plot_widget.paused = self.paused

    def update_window_size(self, 
                    new_size: int):

        self.rt_plot_widget.update_window_size(new_size)

        # updated offset label
        max_offset_current = self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size
        self.window_offset_slider.max_label.setText(f'{max_offset_current}')
        self.window_offset_slider.val_slider.setMaximum(max_offset_current)

        # updates displayed window size
        self.window_size_slider.current_val.setText(\
            f'{self.rt_plot_widget.update_data_dt * self.rt_plot_widget.window_size:.2f}')
    
    def synch_max_window_size(self):

        # update max window size depending on data sample update dt (which might have changed)
        self.window_size_slider.max_label.setText(f'{self.rt_plot_widget.window_buffer_size * self.rt_plot_widget.update_data_dt}')

    def update_window_offset(self, 
                    offset: int):

        self.rt_plot_widget.update_window_offset(offset)

        self.window_offset_slider.current_val.setText(f'{self.rt_plot_widget.window_offset}')

    def toggle_all_visibility(self):
        # Toggle the state of all lines based on the remembered state
        for i, button in enumerate(self.plot_selector.buttons):
            button.setChecked(self.all_lines_visible)

            if self.all_lines_visible:
                self.rt_plot_widget.show_line(i)
                button.setStyleSheet("")  # Reset button style
            else:
                self.rt_plot_widget.hide_line(i)
                button.setStyleSheet("color: darkgray")  # Change button style to dark gray

        # After toggling, remember the new state for the next time
        self.all_lines_visible = not self.all_lines_visible

    def toggle_line_visibility(self, 
                        label):

        for i, legend_label in enumerate(self.rt_plot_widget.labels):
                        
            if label == legend_label:

                checked = self.plot_selector.buttons[i].isChecked()  # Get the new state of the button
                
                if checked:
                    
                    self.rt_plot_widget.show_line(i)

                    self.plot_selector.buttons[i].setStyleSheet("")  # Reset button style

                if not checked:

                    self.rt_plot_widget.hide_line(i)

                    self.plot_selector.buttons[i].setStyleSheet("color: darkgray")  # Change button style to dark gray

    def scroll_legend_names(self, 
                value):
        
        self.settings_scroll_content.setGeometry(0, 
                    -value, 
                    self.settings_scroll_content.width(), 
                    self.settings_scroll_content.height())

class RtPlotWindow():

    def __init__(self, 
            data_dim: int, 
            update_data_dt: float, 
            update_plot_dt: float, 
            window_duration: float, 
            parent: QWidget,
            n_data: int = 1,
            legend_list: List[str] = None,
            base_name: str = "", 
            window_buffer_factor: int = 2, 
            ylabel = ""):

        self.n_data = n_data
        self.data_dim = data_dim

        self.update_data_dt = update_data_dt
        self.update_plot_dt = update_plot_dt    

        self.window_duration = window_duration

        self.base_name = base_name
        self.legend_list = legend_list

        # use a QSplitter to handle resizable width between plot and legend frames
        self.base_frame = QFrame(parent=parent)
        self.base_frame.setFrameShape(QFrame.StyledPanel)
        self.base_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter = QSplitter(Qt.Horizontal)

        # create the plot widget
        self.rt_plot_widget = RtPlotWidget(
            window_duration=self.window_duration,
            update_data_dt=self.update_data_dt,
            update_plot_dt=self.update_plot_dt,
            n_dims=self.data_dim,
            n_data=self.n_data,
            base_name=self.base_name,
            parent=None, 
            window_buffer_factor=window_buffer_factor, 
            legend_list=legend_list, 
            ylabel=ylabel
        )
        # we create the settings widget 
        self.settings_widget = SettingsWidget(rt_plotter=self.rt_plot_widget, 
                            parent=None)
        
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.rt_plot_widget)
        self.splitter.addWidget(self.settings_widget.frame)

        # Set up the layout
        self.splitter_layout = QVBoxLayout()
        self.splitter_layout.addWidget(self.splitter)
        
        self.base_frame.setLayout(self.splitter_layout)        
    
class GridFrameWidget():

    def __init__(self, 
            rows, 
            cols, 
            parent: QWidget = None,
            add_settings_tab = False,
            settings_title = "SETTINGS"):
        
        self.finalized = False

        self.base_frame = QFrame(parent = parent)
        self.base_frame.setFrameShape(QFrame.StyledPanel)
        self.base_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.add_settings_tab = add_settings_tab
        self.settings_frame_layout = None
        self.settings_frame = None
        self.settings_title = settings_title
        self.settings_widget_list = []

        self.rows = rows
        self.cols = cols

        self.settings_splitter = QSplitter(orientation=Qt.Horizontal)
        self.settings_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.row_splitter = QSplitter(Qt.Vertical)
        self.row_splitter.setHandleWidth(5)
        self.row_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.scroll_area_grid = QScrollArea(parent=self.settings_splitter)
        self.scroll_area_grid.setWidgetResizable(True)  # Make the scroll area resizable
        self.scroll_area_grid.setWidget(self.row_splitter)  # Set the frame as the scroll area's widget
        self.scroll_area_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy for scroll area
        self.scroll_area_grid.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        row_layout = QVBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        row_layout.addWidget(self.row_splitter)

        if self.add_settings_tab:
            
            self.settings_frame = QFrame()
            self.settings_frame.setFrameShape(QFrame.StyledPanel)
            self.settings_frame.setContentsMargins(0, 0, 0, 0)
            self.settings_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.settings_frame_layout = QVBoxLayout(self.settings_frame)  # Use QVBoxLayout here
            self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)

            self.settings_title = QLabel(self.settings_title)
            self.settings_title.setAlignment(Qt.AlignTop | Qt.AlignCenter)
            self.settings_frame_layout.addWidget(self.settings_title)

        self.frames = []
        self.col_layouts = []
        self.atomic_layouts = []

        n_rows = 0
        for row in range(rows):
            

            col_frames = []
            atomic_layouts_tmp = []
            
            n_rows = n_rows + 1
            row_frame = QFrame(parent=self.base_frame)
            row_frame.setFrameShape(QFrame.StyledPanel)
            
            col_layout = QHBoxLayout(row_frame)
            col_layout.setContentsMargins(0, 0, 0, 0)
            row_frame.setLayout(col_layout)
            col_splitter = QSplitter(Qt.Horizontal)
            col_splitter.setHandleWidth(2)
            col_layout.addWidget(col_splitter)
            row_layout.addWidget(row_frame)
            self.row_splitter.addWidget(row_frame)

            for col in range(cols):
                
                col_frame = QFrame(parent=row_frame)
                col_frame.setFrameShape(QFrame.StyledPanel)
                atomic_layout = QVBoxLayout()
                atomic_layout.setContentsMargins(0, 0, 0, 0)
                col_frame.setLayout(atomic_layout)
                atomic_layouts_tmp.append(atomic_layout)

                col_layout.addWidget(col_frame)
                col_splitter.addWidget(col_frame)

                col_frames.append(col_frame)

            self.frames.append(col_frames)
            self.col_layouts.append(col_layout)
            self.atomic_layouts.append(atomic_layouts_tmp)
            
            col_frames = [] # reset
            atomic_layouts_tmp = []

        self.settings_splitter.setHandleWidth(1)

        self.settings_splitter.addWidget(self.scroll_area_grid)

        if self.add_settings_tab:
            self.settings_splitter.addWidget(self.settings_frame)
            
        self.settings_splitter_layout = QVBoxLayout()
        self.settings_splitter_layout.addWidget(self.settings_splitter)

        self.base_frame.setLayout(self.settings_splitter_layout) 
        
    def addFrame(self,
            frame: QWidget, 
            row_idx: int, 
            col_idx: int):
        
        if not self.finalized:
            if row_idx < self.rows and \
                col_idx < self.cols:

                self.atomic_layouts[row_idx][col_idx].addWidget(frame)
        else:

            excep = "Can call addFrame only if finalize() was not called yet!"
            
            raise Exception(excep)
        
    def addToSettings(self,
                widget_frame_list: List = []):
        
        if not self.finalized and self.add_settings_tab:

            self.settings_widget_list = widget_frame_list
        
        else:
            
            excep = "Can call addToSettings only before finalize() is called and"  + \
                    "the add_settings_tab was set to True in the constructor!"
            
            raise Exception(excep)
    
    def finalize(self):

        self._finalize_settings_window()

        self.finalized = True

    def _finalize_settings_window(self):
        
        if self.add_settings_tab:
            
            for i in range(len(self.settings_widget_list)):
                
                self.settings_frame_layout.addWidget(self.settings_widget_list[i].base_frame)

            # adding spacer at the end to push all stuff to top
            self.settings_frame_layout.addItem(QSpacerItem(1, 1, 
                                                QSizePolicy.Minimum, QSizePolicy.Expanding))