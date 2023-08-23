import sys
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QHBoxLayout, QFrame
from PyQt5.QtWidgets import QScrollArea, QPushButton, QScrollBar, QSpacerItem, QSizePolicy, QSlider
from PyQt5.QtWidgets import QSplitter, QLabel, QGridLayout

from PyQt5.QtGui import QIcon, QPixmap

import pyqtgraph as pg

from typing import List, Callable

from control_cluster_utils.utilities.sysutils import PathsGetter

import os

class RtPlotWidget(pg.PlotWidget):

    def __init__(self, 
                window_duration: int,
                n_data: int,  
                update_dt: float, 
                base_name: str, 
                legend_list: List[str] = None,
                parent: QWidget = None, 
                xlabel = "t [s]", 
                ylabel = "", 
                window_buffer_factor: int = 2):

        self.warning = "warning"

        super().__init__(title=base_name,
                    parent=parent)
        
        if legend_list is not None and len(legend_list) != n_data:
            
            warning = "[{self.__class__.__name__}]" + f"[{self.warning}]" \
                + f": provided legend list length {len(legend_list)} does not match data dimension {n_data}"
            
            print(warning)

            self.legend_list = None

        self.legend_list = legend_list

        self.x_label = xlabel
        self.y_label = ylabel

        self.paused = False

        self.ntimestamps_per_window = 10
        
        self.n_data = n_data

        self.window_offset = 0

        self.window_size = int(window_duration // update_dt) + 1 # floor division
        self.window_duration = window_duration

        self.window_buffer_factor = window_buffer_factor
        self.window_buffer_size = self.window_buffer_factor * self.window_size
        self.window_buffer_duration = self.window_buffer_factor * self.window_duration

        self.base_name = base_name

        self.update_dt = update_dt

        self.data = np.zeros((self.n_data, self.window_buffer_size))

        self.elapsed_times = [0 for i in range(self.window_buffer_size)]

        self.labels = [] 
        self.lines = []

        self._setup_plot()

        self._init_lines()

        self.update_window_size(self.window_size)

        self.update_timestamps_res(self.ntimestamps_per_window)
        
        self._init_timers()
    
    def _init_lines(self):

        for i in range(0, self.n_data):
            
            if self.legend_list is None:

                label = f"{self.base_name}_{i}"  # generate label for each line

            else:
                
                label = self.legend_list[i]

            self.labels.append(label)

            color = pg.mkColor(self.colors[i])  # Convert intColor to QColor
            color.setAlpha(150)  # Set the alpha value for the color

            pen = pg.mkPen(color=color, 
                    width=2.3)

            self.lines.append(self.plot(self.data[i, :], 
                        pen=pen))
            
    def _setup_plot(self):
        
        # Create a slightly grey background
        self.setBackground('w')  # RGB values for a light grey color

        self.enableAutoRange()
        # Set the alpha of the text to 255 (fully opaque)
        self.getAxis('left').setTextPen(color=(0, 0, 0, 255))  # Black text color with alpha=255
        self.getAxis('bottom').setTextPen(color=(0, 0, 0, 255))  # Black text color with alpha=255

        title_style = {'color': 'k', 'size': '14pt'}
        self.plotItem.setTitle(title=self.base_name, **title_style)
        # Set grid color to black
        self.showGrid(x=True, y=True, alpha=1.0)  # Full opacity for grid lines
        self.getAxis('left').setGrid(255)  # Black grid lines for Y axis
        self.getAxis('bottom').setGrid(255)  # Black grid lines for X axis

        # Define a list of colors for each row
        self.colors = [pg.intColor(i, self.n_data, 255) for i in range(self.n_data)]

        # Add axis labels
        self.plotItem.setLabel('left', self.y_label)
        self.plotItem.setLabel('bottom', self.x_label)

    def _init_timers(self):
        
        self.last_update_time = time.perf_counter()
        self.reference_time = time.perf_counter()
    
    def hide_line(self, 
                index: int):

        self.lines[index].hide() 

    def show_line(self, 
                index: int):

        self.lines[index].show() 
    
    def update(self, 
            new_data: np.ndarray):

        # updates window with new data

        current_time = time.perf_counter()

        if current_time - self.last_update_time >= self.update_dt:

            if not self.paused:

                rolled_data = np.roll(self.data, -1, axis=1)
                self.data = rolled_data
                self.data[:, -1] = new_data.flatten()
                
                for i in range(0, self.data.shape[0]):

                    self.lines[i].setData(self.data[i, :])

                self.elapsed_times = [(current_time - self.reference_time - self.update_dt * i) for i in range(self.window_buffer_size)]
                self.elapsed_times = self.elapsed_times[::-1]

            self._update_timestams_ticks(self.elapsed_times) 

            self.last_update_time = current_time

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
        
class SettingsWidget():

    def __init__(self, 
            rt_plotter: RtPlotWidget,
            parent: QWidget = None
            ):

        self.rt_plot_widget = rt_plotter

        self.frame = QFrame(parent=parent)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.settings_frame_layout = QVBoxLayout(self.frame)  # Use QVBoxLayout here
        self.settings_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.val_frames = []
        self.val_layouts = []
        self.current_vals = []
        self.val_slider_frames = []
        self.val_slider_frame_layouts = []
        self.min_labels = []
        self.max_labels = []
        self.val_sliders = []

        self.clicked_iconpaths = []
        self.iconed_button_frames = []
        self.iconed_button_layouts = []
        self.iconed_buttons = []
        self.icones_buttons = []
        self.pixmaps = []
        self.button_descrs = []
        self.iconpaths = []

        self.paused = False

        self.init_ui()

    def _generate_complex_slider(self, 
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

        self.val_title = QLabel(title)
        current_val = QLabel(init_val_shown)
        current_val.setAlignment(Qt.AlignRight)
        current_val.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")

        val_layout.addWidget(self.val_title, 
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

        self.val_frames.append(val_frame)
        self.val_layouts.append(val_layout)
        self.val_slider_frames.append(val_slider_frame)
        self.val_slider_frame_layouts.append(val_slider_frame_layout)
        self.min_labels.append(min_label)
        self.max_labels.append(max_label)
        self.val_sliders.append(val_slider)
        self.current_vals.append(current_val)

    def _create_iconed_button(self, 
                        parent: QWidget, 
                        icon_basepath: str, 
                        icon: str,
                        descr: str = ""):
        
        iconpath = os.path.join(icon_basepath, 
                                   icon + ".svg")
        
        button_frame = QFrame(parent)
        button_frame.setFrameShape(QFrame.StyledPanel)
        button_frame_color = button_frame.palette().color(self.frame.backgroundRole()).name()
        button_layout = QHBoxLayout(button_frame)  # Use QVBoxLayout here
        button_layout.setContentsMargins(2, 2, 2, 2)

        button_descr = QLabel(descr)
        button_descr.setAlignment(Qt.AlignLeft | Qt.AlignCenter)

        button = QPushButton(button_frame)
        button.setGeometry(100, 100, 100, 50)
        button.setStyleSheet(f"background-color: {button_frame_color};")
        pixmap = QPixmap(iconpath)

        button_icon = QIcon(pixmap)

        button.setIcon(button_icon)
        button.setFixedSize(30, 30)
        button.setIconSize(button.size())
        
        button.clicked.connect(self.change_pause_state)
        
        self.iconpaths.append(iconpath)

        self.iconed_button_frames.append(button_frame)
        self.iconed_button_layouts.append(button_layout)
        self.iconed_buttons.append(button)
        self.icones_buttons.append(button_icon)
        self.pixmaps.append(pixmap)

        self.button_descrs.append(button_descr)

        button_layout.addWidget(button_descr)
        button_layout.addWidget(button)
        self.settings_frame_layout.addWidget(button_frame)
    
    def _create_plot_selector(self):
        
        self.plot_selector_scroll_area = QScrollArea(parent=self.frame)
        self.plot_selector_scroll_area.setWidgetResizable(True)

        # Create a frame for the plot selector
        
        self.selectors_frame = QFrame(self.plot_selector_scroll_area)
        self.selectors_frame.setFrameShape(QFrame.StyledPanel)
        self.selectors_layout = QVBoxLayout(self.selectors_frame)
        self.selectors_layout.setContentsMargins(2, 2, 2, 2)

        # Add title label to the plot selector frame
        plot_selector_title = QLabel("plot selector")
        self.selectors_layout.addWidget(plot_selector_title, 
                                alignment=Qt.AlignHCenter)

        self.legend_buttons = []

        # Add legend buttons to the plot selector frame
        for label in self.rt_plot_widget.labels:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(True)
            button.clicked.connect(lambda checked, l=label: self.toggle_line_visibility(l))
            self.legend_buttons.append(button)
            self.selectors_layout.addWidget(button)
        
        self.plot_selector_scroll_area.setWidget(self.selectors_frame)
        self.settings_frame_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
    def init_ui(self):
        
        paths = PathsGetter()
        icon_basepath = paths.GUI_ICONS_PATH
        self._create_iconed_button(parent=self.frame, 
                            icon_basepath=icon_basepath, 
                            icon="pause", 
                            descr="freeze/unfreeze")
        
        # create slider for window size

        self._generate_complex_slider(title="window size [s]: ", 
                                parent=self.frame, 
                                callback=self.update_window_size, 
                                min_shown=f'{self.rt_plot_widget.update_dt}', 
                                min = 1,
                                max_shown=f'{self.rt_plot_widget.window_buffer_duration}', 
                                max = self.rt_plot_widget.window_buffer_size,
                                init_val_shown =f'{self.rt_plot_widget.update_dt * self.rt_plot_widget.window_size}', 
                                init=self.rt_plot_widget.window_size)

        self._generate_complex_slider(title="window offset [n.samples]: ", 
                                parent=self.frame, 
                                callback=self.update_window_offset, 
                                min_shown=f'{0}', 
                                min = 0,
                                max_shown=f'{self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size}', 
                                max = self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size,
                                init_val_shown =f'{self.rt_plot_widget.window_offset}', 
                                init=self.rt_plot_widget.window_offset)
        
        self._generate_complex_slider(title="n. timestamps: ", 
                                parent=self.frame, 
                                callback=self.update_timestamps_res, 
                                min_shown=f'{1}', 
                                min = 1,
                                max_shown=f'{self.rt_plot_widget.window_size}', 
                                max = self.rt_plot_widget.window_size,
                                init_val_shown =f'{self.rt_plot_widget.window_offset}', 
                                init=self.rt_plot_widget.window_offset)

        self._create_plot_selector()
 
        self.settings_frame_layout.addWidget(self.plot_selector_scroll_area)

    def change_pause_state(self):
        
        self.paused = not self.paused
        
        # if self.paused: 
            
            
        
        # else:

        #     self.iconed_buttons[0].setIcon(self.icones_buttons_clicked[0])

        self.rt_plot_widget.paused = self.paused

    def update_window_size(self, 
                    new_size: int):

        self.rt_plot_widget.update_window_size(new_size)

        # updated offset label
        max_offset_current = self.rt_plot_widget.window_buffer_size - self.rt_plot_widget.window_size
        self.max_labels[1].setText(f'{max_offset_current}')
        self.val_sliders[1].setMaximum(max_offset_current)

        # updates resolution label
        self.max_labels[2].setText(f'{int(self.rt_plot_widget.window_size)}')
        self.val_sliders[2].setMaximum(int(self.rt_plot_widget.window_size))

        # updates displayed window size
        self.current_vals[0].setText(f'{self.rt_plot_widget.update_dt * self.rt_plot_widget.window_size:.2f}')

    def update_window_offset(self, 
                    offset: int):

        self.rt_plot_widget.update_window_offset(offset)

        self.current_vals[1].setText(f'{self.rt_plot_widget.window_offset}')

    def update_timestamps_res(self, 
                    res: int):

        self.rt_plot_widget.update_timestamps_res(res)

        self.current_vals[2].setText(f'{self.rt_plot_widget.ntimestamps_per_window}')

    def toggle_line_visibility(self, label):

        for i, legend_label in enumerate(self.rt_plot_widget.labels):

            if label == legend_label:

                button = self.legend_buttons[i]

                checked = button.isChecked()  # Get the new state of the button
                
                if checked:
                    
                    self.rt_plot_widget.show_line(i)

                    button.setStyleSheet("")  # Reset button style

                if not checked:

                    self.rt_plot_widget.hide_line(i)

                    button.setStyleSheet("color: darkgray")  # Change button style to dark gray

    def scroll_legend_names(self, 
                value):
        
        self.settings_scroll_content.setGeometry(0, 
                    -value, 
                    self.settings_scroll_content.width(), 
                    self.settings_scroll_content.height())

class RtPlotWindow():

    def __init__(self, 
            n_data: int, 
            update_dt: float, 
            window_duration: float, 
            parent: QWidget,
            legend_list: List[str] = None,
            base_name: str = "", 
            window_buffer_factor: int = 2):

        self.n_data = n_data
        self.update_dt = update_dt
        self.window_duration = window_duration

        self.base_name = base_name
        self.legend_list = legend_list

        # use a QSplitter to handle resizable width between plot and legend frames
        self.base_frame = QFrame(parent=parent)
        self.base_frame.setFrameShape(QFrame.StyledPanel)

        self.splitter = QSplitter(Qt.Horizontal)

        # create the plot widget
        self.rt_plot_widget = RtPlotWidget(
            window_duration=self.window_duration,
            update_dt=self.update_dt,
            n_data=self.n_data,
            base_name=self.base_name,
            parent=None, 
            window_buffer_factor=window_buffer_factor, 
            legend_list=legend_list
        )
        # we create the settings widget 
        self.settings_widget = SettingsWidget(rt_plotter=self.rt_plot_widget, 
                            parent=None)
        
        self.splitter.setHandleWidth(0.1)
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
            parent: QWidget = None):
        
        self.base_frame = QFrame(parent = parent)

        self.rows = rows
        self.cols = cols

        row_layout = QVBoxLayout(self.base_frame)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.base_frame.setLayout(row_layout)
        
        row_splitter = QSplitter(Qt.Vertical)
        row_splitter.setHandleWidth(0.1)
        row_layout.addWidget(row_splitter)

        self.frames = []
        self.col_layouts = []
        self.atomic_layouts = []
        
        for row in range(rows):
            
            col_frames = []
            atomic_layouts_tmp = []

            row_frame = QFrame(parent=self.base_frame)
            row_frame.setFrameShape(QFrame.StyledPanel)

            col_layout = QHBoxLayout(row_frame)
            col_layout.setContentsMargins(0, 0, 0, 0)
            row_frame.setLayout(col_layout)
            col_splitter = QSplitter(Qt.Horizontal)
            col_splitter.setHandleWidth(0.2)
            col_layout.addWidget(col_splitter)
            row_layout.addWidget(row_frame)
            row_splitter.addWidget(row_frame)

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

    def addFrame(self,
            frame: QWidget, 
            row_idx: int, 
            col_idx: int):
        
        if row_idx < self.rows and \
            col_idx < self.cols:

            self.atomic_layouts[row_idx][col_idx].addWidget(frame)

class RhcTaskRefWindow():

    def __init__(self, 
            n_contacts: int,
            update_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            parent: QWidget = None):

        self.grid = GridFrameWidget(2, 2, 
                parent=parent)
        
        self.base_frame = self.grid.base_frame

        self.rt_plotters = []

        self.rt_plotters.append(RtPlotWindow(n_data=n_contacts, 
                    update_dt=update_dt, 
                    window_duration=window_duration, 
                    parent=None, 
                    base_name="Contact flags", 
                    window_buffer_factor=window_buffer_factor))
        
        self.rt_plotters.append(RtPlotWindow(n_data=1, 
                    update_dt=update_dt, 
                    window_duration=window_duration, 
                    parent=None, 
                    base_name="Task mode", 
                    window_buffer_factor=window_buffer_factor))
        
        self.rt_plotters.append(RtPlotWindow(n_data=6, 
                    update_dt=update_dt, 
                    window_duration=window_duration, 
                    parent=None, 
                    base_name="Base pose", 
                    window_buffer_factor=window_buffer_factor))
        
        self.rt_plotters.append(RtPlotWindow(n_data=3, 
                    update_dt=update_dt, 
                    window_duration=window_duration, 
                    parent=None, 
                    base_name="CoM pos", 
                    window_buffer_factor=window_buffer_factor))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)

class DataThread(QThread):

    data_updated = pyqtSignal(np.ndarray)

    def __init__(self, 
                update_dt: float, 
                n_data: int):
        
        super().__init__()

        self.update_dt = update_dt

        self.n_data = n_data

    def run(self):

        while True:

            new_data = 2 * (np.random.rand(self.n_data, 1) - np.full((self.n_data, 1), 0.5))

            self.data_updated.emit(new_data)

            time.sleep(self.update_dt)

class RealTimePlotApp(QMainWindow):

    def __init__(self):

        super().__init__()

        n_data = 15
        update_dt_thread = 0.001
        update_dt_debug= 0.05
        window_duration = 10 # [s]

        # main window widget

        self.rt_plot_window = RtPlotWindow(n_data, 
                                update_dt_debug, 
                                window_duration, 
                                parent=self)
        self.setCentralWidget(self.rt_plot_window.base_frame)

        # this thread will handle the update of the plot
        self.data_thread = DataThread(update_dt_thread, 
                                self.rt_plot_window.n_data)
        self.data_thread.data_updated.connect(self.rt_plot_window.rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()

        self.show()
   
if __name__ == "__main__":  

    app = QApplication(sys.argv)

    main_window = RealTimePlotApp()

    sys.exit(app.exec_())

