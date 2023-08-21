import sys
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QHBoxLayout, QFrame
from PyQt5.QtWidgets import QScrollArea, QPushButton, QScrollBar, QSpacerItem, QSizePolicy, QSlider
from PyQt5.QtWidgets import QSplitter, QLabel, QGridLayout

import pyqtgraph as pg

from typing import List, Callable

class RealTimePlotApp(QMainWindow):

    def __init__(self):

        super().__init__()

        n_data = 5
        update_dt = 0.05
        window_duration = 3 # [s]

        # main window widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.rt_plot_widget = RtPlotWindow(n_data, 
                                update_dt, 
                                window_duration, 
                                main_window=self)
        
        self.central_widget.setLayout(self.layout)

        self.show()
   
class RtPlotWidget(pg.PlotWidget):

    def __init__(self, 
                window_duration: int,
                n_data: int,  
                update_dt: float, 
                base_name: str, 
                parent: QWidget = None):

        super().__init__(title=base_name,
                    parent=parent)
        
        self.timestamps_red_factor = 10
        
        self.n_data = n_data

        self.window_offset = 0

        self.window_size = int(window_duration // update_dt) + 1 # floor division
        self.window_duration = window_duration

        self.window_buffer_factor = 2
        self.window_buffer_size = self.window_buffer_factor * self.window_size
        self.window_buffer_duration = self.window_buffer_factor * self.window_duration

        self.base_name = base_name

        self.update_dt = update_dt

        self.data = np.zeros((self.n_data, self.window_buffer_size))

        self.timestamps = np.arange(self.window_buffer_size)

        self.labels = [] 
        self.lines = []

        self._setup_plot()

        self._init_lines()

        self.update_window_size(self.window_size)
        
        self._init_timers()
    
    def _init_lines(self):

        for i in range(0, self.n_data):

            label = f"{self.base_name}_{i}"  # generate label for each line
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
        self.plotItem.setTitle(title="Prova", **title_style)
        # Set grid color to black
        self.showGrid(x=True, y=True, alpha=1.0)  # Full opacity for grid lines
        self.getAxis('left').setGrid(255)  # Black grid lines for Y axis
        self.getAxis('bottom').setGrid(255)  # Black grid lines for X axis

        # Define a list of colors for each row
        self.colors = [pg.intColor(i, self.n_data, 255) for i in range(self.n_data)]

        # Add axis labels
        self.plotItem.setLabel('left', '')
        self.plotItem.setLabel('bottom', 't [s]')

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

            rolled_data = np.roll(self.data, -1, axis=1)
            self.data = rolled_data
            self.data[:, -1] = new_data.flatten()
            
            for i in range(0, self.data.shape[0]):

                self.lines[i].setData(self.data[i, :])

            elapsed_times = [(current_time - self.reference_time - self.update_dt * i) for i in range(self.window_buffer_size)]
            elapsed_times = elapsed_times[::-1]

            self._update_timestams_ticks(elapsed_times) 

            # x_range = (self.window_buffer_size - 1 - self.window_size, 
            #     self.window_buffer_size - 1) 
            # self.setXRange(*x_range)

            self.last_update_time = current_time

    def _update_timestams_ticks(self, 
                        elapsed_times: List[float]):
        
        # display only some of the x-axis timestamps to avoid overlap
        step_size = max(1, len(elapsed_times) // self.timestamps_red_factor) 
        selected_labels = elapsed_times[::step_size]

        x_tick_vals = [i for i, _ in enumerate(elapsed_times) if i % step_size == 0]
        x_tick_names = [f'{t:.1f}s' for t in selected_labels]

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
        
        current_window_buffer_factor = int(self.window_buffer_duration // self.window_size)

        if offset >= current_window_buffer_factor:

            offset = current_window_buffer_factor - 1

        self.window_offset = offset
        
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

    def init_ui(self):
 
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

        self._generate_complex_slider(title="window offset [n.windows]: ", 
                                parent=self.frame, 
                                callback=self.update_window_offset, 
                                min_shown=f'{0}', 
                                min = 0,
                                max_shown=f'{self.rt_plot_widget.window_buffer_factor - 1}', 
                                max = self.rt_plot_widget.window_buffer_factor - 1,
                                init_val_shown =f'{self.rt_plot_widget.window_offset}', 
                                init=self.rt_plot_widget.window_offset)
        
        # Create a frame for the plot selector
        self.plot_selector_frame = QFrame(self.frame)
        self.plot_selector_frame.setFrameShape(QFrame.StyledPanel)
        self.plot_selector_layout = QVBoxLayout(self.plot_selector_frame)
        self.plot_selector_layout.setContentsMargins(2, 2, 2, 2)

        # Add title label to the plot selector frame
        plot_selector_title = QLabel("plot selector")
        self.plot_selector_layout.addWidget(plot_selector_title, 
                                alignment=Qt.AlignHCenter)

        self.legend_buttons = []

        # Add legend buttons to the plot selector frame
        for label in self.rt_plot_widget.labels:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(True)
            button.clicked.connect(lambda checked, l=label: self.toggle_line_visibility(l))
            self.legend_buttons.append(button)
            self.plot_selector_layout.addWidget(button)
        
        self.settings_frame_layout.addWidget(self.plot_selector_frame)
        self.settings_frame_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def update_window_size(self, 
                    new_size: int):

        self.rt_plot_widget.update_window_size(new_size)

        self.max_labels[1].setText(f'{int(self.rt_plot_widget.window_buffer_duration // self.rt_plot_widget.window_size)}')
        self.val_sliders[1].setMaximum(int(self.rt_plot_widget.window_buffer_duration // self.rt_plot_widget.window_size))
        self.current_vals[0].setText(f'{self.rt_plot_widget.update_dt * self.rt_plot_widget.window_size:.2f}')

    def update_window_offset(self, 
                    offset: int):

        self.rt_plot_widget.update_window_offset(offset)

        self.current_vals[1].setText(f'{self.rt_plot_widget.window_offset}')

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

class RtPlotWindow(QWidget):

    def __init__(self, 
            n_data: int, 
            update_dt: float, 
            window_duration: float, 
            main_window: RealTimePlotApp,
            base_name: str = ""):

        super().__init__()

        self.main_window = main_window

        self.n_data = n_data
        self.update_dt = update_dt
        self.window_duration = window_duration

        self.base_name = base_name
        
        # use a QSplitter to handle resizable width between plot and legend frames
        self.splitter_frame = QFrame()
        self.splitter_frame.setFrameShape(QFrame.StyledPanel)

        self.splitter = QSplitter(Qt.Horizontal)

        # create the plot widget
        self.rt_plot_widget = RtPlotWidget(
            window_duration=self.window_duration,
            update_dt=self.update_dt,
            n_data=self.n_data,
            base_name=self.base_name,
            parent=None
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
        self.splitter_frame.setLayout(self.splitter_layout)
        
        # Set the central widget of the main window
        self.main_window.setCentralWidget(self.splitter_frame)

        # this thread will handle the update of the plot
        self.data_thread = DataThread(self.rt_plot_widget, 
                                self.n_data)
        self.data_thread.data_updated.connect(self.rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()
        
class DataThread(QThread):

    data_updated = pyqtSignal(np.ndarray)

    def __init__(self, 
                rt_plotter: RtPlotWidget, 
                n_data: int):
        
        super().__init__()

        self.rt_plot_widget = rt_plotter

        self.n_data = n_data

    def run(self):

        while True:

            new_data = 2 * (np.random.rand(self.n_data, 1) - np.full((self.n_data, 1), 0.5))

            self.data_updated.emit(new_data)

            time.sleep(self.rt_plot_widget.update_dt)

if __name__ == "__main__":  

    app = QApplication(sys.argv)

    main_window = RealTimePlotApp()

    sys.exit(app.exec_())

