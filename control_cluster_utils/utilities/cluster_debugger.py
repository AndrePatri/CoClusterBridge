from PyQt5.QtCore import QThread, pyqtSignal, Qt

from control_cluster_utils.utilities.plot_utils import RtPlotWindow

from PyQt5.QtWidgets import QApplication, QMainWindow

import sys

import torch
import numpy as np

import time

from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs
from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.defs import cluster_size_name, n_contacts_name

class SharedDataThread(QThread):

    data_updated = pyqtSignal(np.ndarray)

    def __init__(self, 
                update_dt: float, 
                index: int = 0,
                verbose = True):
        
        super().__init__()

        self.update_dt = update_dt

        
        self.verbose = verbose
        self.wait_amount = 0.05
        self.dtype = torch.float32
        self.cluster_size = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        self.n_contacts = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=n_contacts_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        
        self.cluster_size.attach()
        self.n_contacts.attach()
        
        cluster_size = self.cluster_size.tensor_view[0, 0].item()
        n_contacts = self.n_contacts.tensor_view[0, 0].item()

        # view of rhc references
        self.rhc_refs = RhcTaskRefs( 
            cluster_size=cluster_size,
            n_contacts=n_contacts,
            index=index,
            q_remapping=None,
            dtype=self.dtype, 
            verbose=verbose)        
        
        self.n_data = self.rhc_refs.phase_id.is_contact.numpy().shape[1]
              
    def run(self):

        while True:

            try: 

                new_data = self.rhc_refs.phase_id.is_contact.numpy()

                self.data_updated.emit(new_data)

                time.sleep(self.update_dt)
            
            except KeyboardInterrupt:

                break

class RtClusterDebugger(QMainWindow):

    def __init__(self, 
                update_dt: float = 0.05):

        super().__init__()

        update_dt = 0.01
        window_length = 10 # [s]
        window_buffer_factor = 2
        # main window widget

        self.data_thread = SharedDataThread(update_dt)

        self.rt_plot_window = RtPlotWindow(n_data=self.data_thread.n_data, 
                                update_dt=update_dt, 
                                window_duration=window_length, 
                                window_buffer_factor=window_buffer_factor,
                                parent=self, 
                                base_name="RhcTaskRef")
        
        self.setCentralWidget(self.rt_plot_window.base_frame)

        self.show()

        # this thread will handle the update of the plot
        
        self.data_thread.data_updated.connect(self.rt_plot_window.rt_plot_widget.update,
                                        Qt.QueuedConnection)
        self.data_thread.start()


if __name__ == "__main__":  

    app = QApplication(sys.argv)

    main_window = RtClusterDebugger()

    sys.exit(app.exec_())