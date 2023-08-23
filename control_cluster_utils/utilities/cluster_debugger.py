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

    rhc_task_contacts = pyqtSignal(np.ndarray)
    rhc_task_mode = pyqtSignal(int)
    rhc_task_base_pose = pyqtSignal(np.ndarray)
    rhc_task_com_pos = pyqtSignal(np.ndarray)

    rhc_cmd_q = pyqtSignal(np.ndarray)
    rhc_cmd_v = pyqtSignal(np.ndarray)
    rhc_cmd_eff = pyqtSignal(np.ndarray)
    rhc_info = pyqtSignal(np.ndarray)

    rhc_state_root_p = pyqtSignal(np.ndarray)
    rhc_state_root_q = pyqtSignal(np.ndarray)
    rhc_state_root_v = pyqtSignal(np.ndarray)
    rhc_state_root_omega = pyqtSignal(np.ndarray)
    rhc_state_jnt_q = pyqtSignal(np.ndarray)
    rhc_state_jnt_v = pyqtSignal(np.ndarray)

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

        self._init_shared_data()
                
        self.initialized = True

    def terminate(self):

        self._terminate = True

    def _emit_data(self):
        
        t = time.perf_counter()

        self.rhc_task_contacts.emit(self.rhc_task_refs[self._cluster_index].phase_id.get_contacts().numpy())
        self.rhc_task_mode.emit(self.rhc_task_refs[self._cluster_index].phase_id.get_phase_id())
        self.rhc_task_base_pose.emit(self.rhc_task_refs[self._cluster_index].base_pose.get_pose().numpy())
        self.rhc_task_com_pos.emit(self.rhc_task_refs[self._cluster_index].com_pos.get_com_pos().numpy())

        update_duration = time.perf_counter() - t # compensate for emit time

        time.sleep(self.update_dt - update_duration)
        
    def run(self):

        if self.initialized:
            
            while True:
                
                self._emit_data()
    
    def _init_shared_data(self):

        self.wait_amount = 0.05
        self.dtype = torch.float32
        
        # getting info
        self.cluster_size_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        self.cluster_size_clnt.attach()
        self.n_contacts_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=n_contacts_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        self.n_contacts_clnt.attach()
        self.jnt_number_clnt = SharedMemClient(n_rows=1, n_cols=1,
                                        name=jnt_number_client_name(), 
                                        dtype=torch.int64, 
                                        wait_amount=self.wait_amount, 
                                        verbose=self.verbose)
        self.jnt_number_clnt.attach()
        self.jnt_names_clnt = SharedStringArray(length=self.jnt_number_clnt.tensor_view[0, 0].item(), 
                                    name=jnt_names_client_name(), 
                                    is_server=False, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        self.jnt_names_clnt.start()
        self.add_data_length_clnt = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=additional_data_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        self.add_data_length_clnt.attach()

        self.cluster_size = self.cluster_size_clnt.tensor_view[0, 0].item()
        self.n_contacts = self.n_contacts_clnt.tensor_view[0, 0].item()
        self.jnt_names = self.jnt_names_clnt.read()
        self.jnt_number = self.jnt_number_clnt.tensor_view[0, 0].item()
        self.add_data_length = self.add_data_length_clnt.tensor_view[0, 0].item()

        # view of rhc references
        for i in range(0, self.cluster_size):

            self.rhc_task_refs.append(RhcTaskRefs( 
                cluster_size=self.cluster_size,
                n_contacts=self.n_contacts,
                index=i,
                q_remapping=None,
                dtype=self.dtype, 
                verbose=self.verbose))

            self.rhc_cmd.append(RobotCmds(n_dofs=self.jnt_number, 
                                    cluster_size=self.cluster_size, 
                                    index=i, 
                                    jnt_remapping=None, # we see everything as seen on the simulator side 
                                    add_info_size=self.add_data_length, 
                                    dtype=self.dtype, 
                                    verbose=self.verbose))

            self.rhc_state.append(RobotState(n_dofs=self.jnt_number, 
                                    cluster_size=self.cluster_size, 
                                    index=i, 
                                    jnt_remapping=None, 
                                    q_remapping=None, 
                                    dtype=self.dtype, 
                                    verbose=self.verbose))

    def update_cluster_idx(self, 
                    idx: int):

        if idx < self.cluster_size:

            self._cluster_index = idx

class RtClusterDebugger(QMainWindow):

    cluster_selector = pyqtSignal(int)

    def __init__(self, 
                update_dt: float = 0.05):

        self.app = QApplication(sys.argv)

        super().__init__()

        update_dt = 0.01
        window_length = 10 # [s]
        window_buffer_factor = 2
        # main window widget

        self.data_thread = SharedDataThread(update_dt)

        self.rhc_task_plotter = RhcTaskRefWindow(n_contacts=self.data_thread.n_contacts, 
                                    update_dt=update_dt, 
                                    window_duration=window_length, 
                                    window_buffer_factor=window_buffer_factor, 
                                    parent=None)
        
        self.setCentralWidget(self.rhc_task_plotter.base_frame)

        # this thread will handle the update of the plot
        
        self.cluster_selector.connect(self.data_thread.update_cluster_idx, 
                                    Qt.QueuedConnection)
        
        self.data_thread.rhc_task_contacts.connect(self.rhc_task_plotter.rt_plotters[0].rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.rhc_task_mode.connect(self.rhc_task_plotter.rt_plotters[1].rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.rhc_task_base_pose.connect(self.rhc_task_plotter.rt_plotters[2].rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.rhc_task_com_pos.connect(self.rhc_task_plotter.rt_plotters[3].rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()

        self.update_cluster_idx(1)

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

    main_window = RtClusterDebugger()

    main_window.run()
