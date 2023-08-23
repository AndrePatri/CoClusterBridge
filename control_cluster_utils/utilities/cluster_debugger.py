from PyQt5.QtCore import QThread, pyqtSignal, Qt

from control_cluster_utils.utilities.plot_utils import RhcTaskRefWindow

from PyQt5.QtWidgets import QApplication, QMainWindow

import sys

import torch
import numpy as np

import time

from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs, RobotCmds, RobotState
from control_cluster_utils.utilities.shared_mem import SharedMemClient, SharedStringArray
from control_cluster_utils.utilities.defs import cluster_size_name, n_contacts_name
from control_cluster_utils.utilities.defs import jnt_names_client_name, jnt_number_client_name
from control_cluster_utils.utilities.defs import additional_data_name

class SharedDataThread(QThread):

    trigger_update = pyqtSignal(int)

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

        self.trigger_update.emit(self._cluster_index)
        
        update_duration = time.perf_counter() - t # compensate for emit time

        actual_sleep = self.update_dt - update_duration

        if (actual_sleep > 0):

            time.sleep(actual_sleep)
        
    def run(self):

        if self.initialized:
            
            while True:
                
                self._trigger_update()

    def update_cluster_idx(self, 
                    idx: int):

        if idx < self.cluster_size:

            self._cluster_index = idx

class RtClusterDebugger(QMainWindow):

    cluster_selector = pyqtSignal(int)

    def __init__(self, 
                update_dt: float = 0.05, 
                verbose: bool = False):

        self.app = QApplication(sys.argv)

        super().__init__()

        update_dt = 0.02
        window_length = 10 # [s]
        window_buffer_factor = 2
        # main window widget

        self.data_thread = SharedDataThread(update_dt)

        self.rhc_task_plotter = RhcTaskRefWindow(update_dt=update_dt, 
                                    window_duration=window_length, 
                                    window_buffer_factor=window_buffer_factor, 
                                    parent=None, 
                                    verbose = verbose)
        
        self.setCentralWidget(self.rhc_task_plotter.base_frame)
        
        self.cluster_selector.connect(self.data_thread.update_cluster_idx, 
                                    Qt.QueuedConnection)
        
        self.data_thread.trigger_update.connect(self.rhc_task_plotter.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()

        # self.update_cluster_idx(1)

        self.show()

    def update_cluster_idx(self, 
                        idx: int):

        self.cluster_selector.emit(idx)

    def run(self):

        self.app.exec_()

    def __del__(self):

        self.terminate()

    def terminate(self):

        self.data_thread.terminate()

if __name__ == "__main__":  

    main_window = RtClusterDebugger(verbose=True)

    main_window.run()
