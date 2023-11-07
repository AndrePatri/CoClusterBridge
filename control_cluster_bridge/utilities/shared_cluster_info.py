from control_cluster_bridge.utilities.shared_mem import SharedMemSrvr, SharedMemClient, SharedStringArray

import torch

class SharedClusterInfo:
      
    def __init__(self, 
                name = "", 
                is_client = False):
        
        self._terminate = False

        self.is_client = is_client

        self.init = None
        
        self.shared_cluster_data = None

        self.namespace = name 

        if self.is_client:
             
            self.shared_cluster_data = SharedMemClient(
                                        name="SharedClusterInfo",
                                        namespace=self.namespace, 
                                        dtype=torch.float32, 
                                        verbose=True)
            
        else:
            
            self.init = ["solve_time_cluster_client [s]", 
                        "controllers active [>0 True]"]

            self.shared_cluster_datanames = SharedStringArray(len(self.init), 
                                                "SharedClusterInfoNames", 
                                                is_server=not self.is_client, 
                                                namespace=self.namespace, 
                                                verbose=True)
            
            self.shared_cluster_data = SharedMemSrvr(n_rows=1, 
                                        n_cols=len(self.init), 
                                        name="SharedClusterInfo",
                                        namespace=self.namespace, 
                                        dtype=torch.float32)
    
    def start(self):
    
        if self.is_client:
        
            self.shared_cluster_data.attach() 

            self.shared_cluster_datanames = SharedStringArray(len(self.shared_cluster_data.tensor_view[0, :]), 
                                        "SharedClusterInfoNames", 
                                        is_server=not self.is_client, 
                                        namespace=self.namespace, 
                                        verbose=True)
            

            self.shared_cluster_datanames.start()

        else:

            self.shared_cluster_datanames.start(self.init) 

            self.shared_cluster_data.start() 

            # write static information
            self.shared_cluster_data.tensor_view[0, 0] = -1.0

            self.shared_cluster_data.tensor_view[0, 1] = -1.0

    def get_names(self):
    
        return self.shared_cluster_datanames.read()
    
    def update(self, 
               solve_time, 
               controllers_up):

        # write runtime information

        self.shared_cluster_data.tensor_view[0, 0] = solve_time  

        self.shared_cluster_data.tensor_view[0, 1] = controllers_up       

    def get_solve_time(self):

        return self.shared_cluster_data.tensor_view[0, 0]
    
    def are_controllers_up(self):

        return self.shared_cluster_data.tensor_view[0, 1]
    
    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self.shared_cluster_datanames.terminate()

            self.shared_cluster_data.terminate()

    def __del__(self):
        
        self.terminate()
