import torch

from abc import ABC

from typing import TypeVar

class RobotClusterState:

    class RootStates:

        def __init__(self, 
                    cluster_size: int = 1, 
                    device: torch.device = torch.device("cpu"), 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device

            self.p = torch.zeros((cluster_size, 3), device = self._device, dtype=self.dtype) # floating base positions
            self.q = torch.zeros((cluster_size, 4), device = self._device, dtype=self.dtype) # floating base orientation (quaternion)
            self.v = torch.zeros((cluster_size , 3), device = self._device, dtype=self.dtype) # floating base linear vel
            self.omega = torch.zeros((cluster_size , 3), device = self._device, dtype=self.dtype) # floating base linear vel

    class JntStates:

        def __init__(self, 
                    n_dofs: int, 
                    cluster_size: int = 1, 
                    device: str = "cpu", 
                    dtype = torch.float32):

            self.dtype = dtype

            self._device = device

            self.q = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint positions
            self.v = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint velocities
            self.a = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations
            self.eff = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"), 
                dtype: torch.dtype = torch.float32):
        
        self.dtype = dtype

        self.backend = "torch" # forcing torch backend

        self.device = device
        
        self.cluster_size = cluster_size
        self.n_dofs = n_dofs
        
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.root_state = self.RootStates(cluster_size = cluster_size, 
                                        device = self.device, 
                                        dtype = self.dtype)

        self.jnt_state = self.JntStates(n_dofs = n_dofs, 
                                        cluster_size = cluster_size, 
                                        device = self.device, 
                                        dtype = self.dtype)
        
        self.aggregate_view = torch.cat([
            self.root_state.p,
            self.root_state.q,
            self.root_state.v,
            self.root_state.omega,
            self.jnt_state.q,
            self.jnt_state.v
        ], dim=1)

    def update(self, other_cmd: 'RobotClusterState', 
               synch = False) -> None:
        
        if self.cluster_size != other_cmd.cluster_size or \
                self.n_dofs != other_cmd.n_dofs:
            
            exception = f"[{self.__class__.__name__}]"  + f"[{self._exception}]" +  f"[{self.update.__name__}]: " + \
                        f"dimensions of provided cluster state {other_cmd._cluster_size} x {other_cmd._n_dofs} " + \
                        f"do not match {self._cluster_size} x {self._n_dofs}"
            
            raise ValueError(exception)

        self.aggregate_view.copy_(other_cmd.aggregate_view, 
                                non_blocking=True) # non-blocking -> we need to synchronize with 
        # before accessing the copied data

        if synch: 
            
            torch.cuda.synchronize()

class RobotClusterCmd:

    class JntCmd:

        def __init__(self, 
                    n_dofs: int, 
                    cluster_size: int = 1, 
                    device: torch.device = torch.device("cpu"), 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device
           
            self._cluster_size = cluster_size
            self._n_dofs = n_dofs

            self.q = torch.zeros((cluster_size, n_dofs), 
                                device = self._device, 
                                dtype=self.dtype) # joint positions
            self.v = torch.zeros((cluster_size, n_dofs),
                                device = self._device, 
                                dtype=self.dtype) # joint velocities
        #     self.a = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations
            self.eff = torch.zeros((cluster_size, n_dofs), 
                                device = self._device, 
                                dtype=self.dtype) # joint accelerations
            
            self._status = "status"
            self._info = "info"
            self._warning = "warning"
            self._exception = "exception"

    class RhcInfo:

        def __init__(self, 
                    cluster_size: int = 1, 
                    add_data_size: int = 1,
                    device: torch.device = torch.device("cpu"), 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device
            
            self._add_data_size = add_data_size

            self.data = torch.zeros((cluster_size, self._add_data_size), 
                                device = self._device,
                                dtype=self.dtype)

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"),  
                dtype: torch.dtype = torch.float32, 
                add_data_size: int = None):

        self.n_dofs = n_dofs
        self.cluster_size = cluster_size
    
        self.dtype = dtype

        self.backend = "torch" # forcing torch backend

        self.device = device

        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.jnt_cmd = self.JntCmd(n_dofs = self.n_dofs, 
                                cluster_size = self.cluster_size, 
                                device = self.device, 
                                dtype = self.dtype)
        
        if add_data_size is not None:
            
                self.rhc_info = self.RhcInfo(cluster_size=self.cluster_size, 
                                        device=self.device, 
                                        dtype=self.dtype, 
                                        add_data_size = add_data_size)
                
                self.aggregate_view = torch.cat([
                self.jnt_cmd.q,
                self.jnt_cmd.v,
                self.jnt_cmd.eff,
                self.rhc_info.data,
                ], dim=1)
        
        else:
             
             self.aggregate_view = torch.cat([
                self.jnt_cmd.q,
                self.jnt_cmd.v,
                self.jnt_cmd.eff,
                ], dim=1)
        
    def update(self, other_cmd: 'RobotClusterCmd', 
               synch = False) -> None:
        
        if self._cluster_size != other_cmd._cluster_size or \
                self._n_dofs != other_cmd._n_dofs:
            
            exception = f"[{self.__class__.__name__}]"  + f"[{self._exception}]" +  f"[{self.update.__name__}]: " + \
                        f"dimensions of provided cluster command {other_cmd._cluster_size} x {other_cmd._n_dofs} " + \
                        f"do not match {self._cluster_size} x {self._n_dofs}"
            
            raise ValueError(exception)

        self.aggregate_view.copy_(other_cmd.aggregate_view, 
                                non_blocking=True) # non-blocking -> we need to synchronize with 
        # before accessing the copied data

        if synch: 
            
            torch.cuda.synchronize()

class Action(ABC):
        
        pass

ActionChild = TypeVar('ActionChild', bound='Action')