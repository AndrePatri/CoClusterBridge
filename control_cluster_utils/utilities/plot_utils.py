import sys
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QHBoxLayout, QFrame
from PyQt5.QtWidgets import QScrollArea, QPushButton, QScrollBar, QSpacerItem, QSizePolicy, QSlider
from PyQt5.QtWidgets import QSplitter, QLabel, QGridLayout

import pyqtgraph as pg

from typing import List

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
        
class RtPlotWidget(pg.PlotWidget):

    def __init__(self, 
                window_duration: int,
                n_data: int,  
                update_dt: float, 
                base_name: str, 
                plot_frame: QFrame):

        super().__init__(title=base_name)

        self.timestamps_red_factor = 10

        self.plot_frame = plot_frame  # Store the reference to the plot frame
        
        self.n_data = n_data

        self.window_size = int(window_duration // update_dt) # floor division
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

            x_range = (self.window_buffer_size - 1 - self.window_size, 
                self.window_buffer_size - 1) 
            self.setXRange(*x_range)

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

        x_range = (self.window_buffer_size - 1 - self.window_size, 
            self.window_buffer_size - 1) 
        
        self.setXRange(*x_range)
        
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

class SettingsWidget(QWidget):

    def __init__(self, 
            rt_plotter: RtPlotWidget,
            parent_frame: QFrame = None
            ):
        
        super().__init__(parent_frame)

        self.rt_plot_widget = rt_plotter

        self.parent_frame = parent_frame

        self.init_ui()

    def init_ui(self):
        
        self.settings_layout = QVBoxLayout(self.parent_frame)
        self.settings_layout.setContentsMargins(5, 5, 5, 5)

        # create a scroll area for the legend names
        self.settings_scroll_area = QScrollArea(self.parent_frame)
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_content = QWidget(self.settings_scroll_area)
        self.settings_scroll_layout = QVBoxLayout(self.settings_scroll_content)
        
        # create slider for window size
        self.window_size_frame = QFrame(self.settings_scroll_area)
        self.window_size_frame.setFrameShape(QFrame.StyledPanel)
        self.window_size_layout = QVBoxLayout(self.window_size_frame)  # Use QVBoxLayout here
        self.window_size_layout.setContentsMargins(5, 5, 5, 5)

        self.window_size_title = QLabel("window size [s]")
        self.window_size_layout.addWidget(self.window_size_title, alignment=Qt.AlignHCenter)

        self.window_size_slider_frame = QFrame(self.settings_scroll_area)
        self.window_size_slider_frame.setFrameShape(QFrame.StyledPanel)
        self.window_size_slider_frame_layout = QHBoxLayout(self.window_size_slider_frame)  # Use QHBoxLayout here
        self.window_size_slider_frame_layout.setContentsMargins(5, 5, 5, 5)

        self.min_label = QLabel(f'{self.rt_plot_widget.update_dt}')
        self.min_label.setAlignment(Qt.AlignCenter)
        self.min_label.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")
        self.window_size_slider_frame_layout.addWidget(self.min_label)

        self.window_size_slider = QSlider(Qt.Horizontal)
        self.window_size_slider.setMinimum(1)
        self.window_size_slider.setMaximum(self.rt_plot_widget.window_buffer_size)
        self.window_size_slider.setValue(self.rt_plot_widget.window_size)
        self.window_size_slider.valueChanged.connect(self.update_window_size)
        self.window_size_slider_frame_layout.addWidget(self.window_size_slider)

        self.max_label = QLabel(f'{self.rt_plot_widget.window_buffer_duration}')
        self.max_label.setAlignment(Qt.AlignCenter)
        self.max_label.setStyleSheet("border: 1px solid gray; background-color: white; border-radius: 4px;")
        self.window_size_slider_frame_layout.addWidget(self.max_label)

        self.window_size_layout.addWidget(self.window_size_slider_frame)

        # we add this window to the settings one
        self.settings_layout.addWidget(self.window_size_frame)

        # Create a frame for the plot selector
        self.plot_selector_frame = QFrame(self.settings_scroll_area)
        self.plot_selector_frame.setFrameShape(QFrame.StyledPanel)
        self.plot_selector_layout = QVBoxLayout(self.plot_selector_frame)
        self.plot_selector_layout.setContentsMargins(5, 5, 5, 5)

        # Add title label to the plot selector frame
        plot_selector_title = QLabel("plot selector")
        self.plot_selector_layout.addWidget(plot_selector_title, alignment=Qt.AlignHCenter)

        self.legend_buttons = []

        # Add legend buttons to the plot selector frame
        for label in self.rt_plot_widget.labels:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(True)
            button.clicked.connect(lambda checked, l=label: self.toggle_line_visibility(l))
            self.legend_buttons.append(button)
            self.plot_selector_layout.addWidget(button)

        # Add the plot selector frame to the main layout
        self.settings_layout.addWidget(self.plot_selector_frame)

        # Add another spacer to push the buttons upwards
        # bottom_spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.settings_scroll_layout.addItem(bottom_spacer)

        # Add a vertical slider for scrolling through legend names
        self.legend_scroll_slider = QScrollBar(Qt.Vertical, self.plot_selector_frame)
        self.legend_scroll_slider.valueChanged.connect(self.scroll_legend_names)
        self.settings_scroll_content.setLayout(self.settings_scroll_layout)
        self.settings_scroll_area.setWidget(self.settings_scroll_content)
        self.settings_scroll_area.setVerticalScrollBar(self.legend_scroll_slider)

        self.parent_frame.setLayout(self.settings_layout)

    def update_window_size(self, new_size):

        self.rt_plot_widget.update_window_size(new_size)

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

        # create the plot frame
        self.plot_frame = QFrame(main_window.central_widget) # attach to main window
        self.plot_frame.setFrameShape(QFrame.StyledPanel)
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_layout.setContentsMargins(3, 3, 3, 3)

        self.rt_plot_widget = RtPlotWidget(
            window_duration=self.window_duration,
            update_dt=self.update_dt,
            n_data=self.n_data,
            base_name=self.base_name,
            plot_frame=self.plot_frame  # Pass the plot_frame to RtPlotWidget
        )

        # add the actual plot to the frame
        self.plot_layout.addWidget(self.rt_plot_widget)
        self.plot_frame.setLayout(self.plot_layout)

        # create a frame for the legend
        self.settings_frame = QFrame(main_window.central_widget)
        self.settings_frame.setFrameShape(QFrame.StyledPanel)
        
        # we add the settings widget 
        self.settings_window = SettingsWidget(rt_plotter=self.rt_plot_widget, 
                                        parent_frame=self.settings_frame)
        
        # add the plot and legend frames to the main layout
        main_window.layout.addWidget(self.plot_frame)
        main_window.layout.addWidget(self.settings_frame, 
                        alignment=Qt.AlignTop)  # align the legend frame to the top

        # use a QSplitter to handle resizable width between plot and legend frames
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(0.1)
        self.splitter.addWidget(self.plot_frame)
        self.splitter.addWidget(self.settings_frame)

        main_window.layout.addWidget(self.splitter)  # add the splitter to the main layout

        # this thread will handle the update of the plot
        self.data_thread = DataThread(self.rt_plot_widget, 
                                self.n_data)
        self.data_thread.data_updated.connect(self.rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()
        
if __name__ == "__main__":  

    app = QApplication(sys.argv)

    main_window = RealTimePlotApp()
    main_window.show()

    sys.exit(app.exec_())

