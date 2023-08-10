import torch

from abc import ABC

from typing import TypeVar

from control_cluster_utils.utilities.shared_mem import SharedMemSrvr

from control_cluster_utils.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_utils.utilities.defs import states_name, cmds_name

class RobotClusterState:

    class RootStates:

        def __init__(self, 
                cluster_aggregate: torch.Tensor):
            
            self.p = None # floating base positions
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base linear vel

            self.cluster_size = cluster_aggregate.shape[0]

            self.assign_views(cluster_aggregate, "p")
            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")
            self.assign_views(cluster_aggregate, "omega")

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "p":

                # (can only make views of contigous memory)

                offset = 0
                
                self.p = cluster_aggregate[:, offset:(offset + 3)].view(self.cluster_size, 
                                                                    3)
                
            if varname == "q":

                offset = 3
                
                self.q = cluster_aggregate[:, offset:(offset + 4)].view(self.cluster_size, 
                                                                    4)
            
            if varname == "v":

                offset = 7
                
                self.v = cluster_aggregate[:, offset:(offset + 3)].view(self.cluster_size, 
                                                                    3)
                
            if varname == "omega":

                offset = 10
                
                self.omega = cluster_aggregate[:, offset:(offset + 3)].view(self.cluster_size, 
                                                                    3)
    class JntStates:

        def __init__(self, 
                    cluster_aggregate: torch.Tensor,
                    n_dofs: int):
        
            self.n_dofs = n_dofs

            self.cluster_size = cluster_aggregate.shape[0]
        
            self.q = None # joint positions
            self.v = None # joint velocities

            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "q":

                # (can only make views of contigous memory)

                offset = 13
                
                self.q = cluster_aggregate[:, offset:(offset + self.n_dofs)].view(self.cluster_size, 
                                                self.n_dofs)
                
            if varname == "v":
                
                offset = 13 + self.n_dofs

                self.v = cluster_aggregate[:, offset:(offset + self.n_dofs)].view(self.cluster_size, 
                                                self.n_dofs)
                
    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"), 
                dtype: torch.dtype = torch.float32):
        
        self.dtype = dtype

        self.backend = "torch" # forcing torch backend
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_dofs = n_dofs
        cluster_aggregate_columnsize = aggregate_state_size(self.n_dofs)

        self.cluster_aggregate = torch.zeros(
                                    (self.cluster_size, 
                                        cluster_aggregate_columnsize 
                                    ), 
                                    dtype=self.dtype, 
                                    device=self.device)

        # views of cluster_aggregate
        self.root_state = self.RootStates(self.cluster_aggregate) 
        self.jnt_state = self.JntStates(self.cluster_aggregate, 
                                    n_dofs)
        
        # this creates a shared memory block of the right size for the state
        # and a corresponding view of it
        self.shared_memman = SharedMemSrvr(self.cluster_size, 
                                    cluster_aggregate_columnsize, 
                                    states_name(), 
                                    self.dtype) # the client will wait until
        # the memory becomes available
        
        

    def synch(self):

        # synchs root_state and jnt_state (which will normally live on GPU)
        # with the shared state data using the aggregate view (normally on CPU)

        # this requires a COPY FROM GPU TO CPU
        # (better to use the aggregate to exploit parallelization)

        self.shared_memman.tensor_view[:, :] = self.cluster_aggregate.cpu()

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

class RobotClusterCmd:

    class JntCmd:

        def __init__(self,
                    cluster_aggregate: torch.Tensor, 
                    n_dofs: int):
            
            self._cluster_size = cluster_aggregate.shape[0]

            self._n_dofs = n_dofs

            self.q = None # joint positions
            self.v = None # joint velocities
            self.eff = None # joint accelerations
            
            self._status = "status"
            self._info = "info"
            self._warning = "warning"
            self._exception = "exception"

            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")
            self.assign_views(cluster_aggregate, "eff")

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "q":
                
                # can only make views of contigous memory
                self.q = cluster_aggregate[:, 0:self._n_dofs].view(self._cluster_size, 
                                                self._n_dofs)
                
            if varname == "v":
                
                offset = self._n_dofs
                self.v = cluster_aggregate[:, offset:(offset + self._n_dofs)].view(self._cluster_size, 
                                                self._n_dofs)
            
            if varname == "eff":
                
                offset = 2 * self._n_dofs
                self.eff = cluster_aggregate[:, offset:(offset + self._n_dofs)].view(self._cluster_size, 
                                                self._n_dofs)
                
    class RhcInfo:

        def __init__(self,
                    cluster_aggregate: torch.Tensor, 
                    add_data_size: int, 
                    n_dofs: int):
            
            self.add_data_size = add_data_size
            self.n_dofs = n_dofs

            self._cluster_size = cluster_aggregate.shape[0]

            self.info = None

            self.assign_views(cluster_aggregate, "info")

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "info":
                
                offset = 3 * self.n_dofs
                self.info = cluster_aggregate[:, 
                                offset:(offset + self.add_data_size)].view(self._cluster_size, 
                                self.add_data_size)
                
    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"),  
                dtype: torch.dtype = torch.float32, 
                add_data_size: int = None):
    
        self.dtype = dtype

        self.backend = "torch" # forcing torch backen
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_dofs = n_dofs
        
        cluster_aggregate_columnsize = -1

        if add_data_size is not None:
            
            cluster_aggregate_columnsize = aggregate_cmd_size(self.n_dofs, 
                                                        add_data_size)
            self.cluster_aggregate = torch.zeros(
                                        (self.cluster_size, 
                                           cluster_aggregate_columnsize 
                                        ), 
                                        dtype=self.dtype, 
                                        device=self.device)
            
            self.rhc_info = self.RhcInfo(self.cluster_aggregate,
                                    add_data_size, 
                                    self.n_dofs)
            
        else:
            
            cluster_aggregate_columnsize = aggregate_cmd_size(self.n_dofs, 
                                                            0)
                                                        
            self.cluster_aggregate = torch.zeros(
                                        (self.cluster_size,
                                            cluster_aggregate_columnsize
                                        ), 
                                        dtype=self.dtype, 
                                        device=self.device)
        
        self.jnt_cmd = self.JntCmd(self.cluster_aggregate,
                                    n_dofs = self.n_dofs)
        
        # this creates a shared memory block of the right size for the cmds
        # and a corresponding view of it
        self.shared_memman = SharedMemSrvr(self.cluster_size, 
                                    cluster_aggregate_columnsize, 
                                    cmds_name(), 
                                    self.dtype) # the client will wait until
        # the memory becomes available

    def synch(self):

        # synchs jnt_cmd and rhc_info (which will normally live on GPU)
        # with the shared cmd data using the aggregate view (normally on CPU)

        # this requires a COPY FROM CPU TO GPU
        # (better to use the aggregate to exploit parallelization)

        self.cluster_aggregate[:, :] = self.shared_memman.tensor_view.cuda()

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated