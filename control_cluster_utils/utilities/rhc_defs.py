import torch

from typing import TypeVar

from control_cluster_utils.utilities.shared_mem import SharedMemClient

class RobotState:

    class RootState:

        def __init__(self, 
                    mem_manager: SharedMemClient):
            
            self.p = None # floating base position
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base angular vel

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "p")
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "omega")

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "p":
                
                self.p = mem_manager.create_tensor_view(index=0, 
                                        length=3)

            if varname == "q":
                
                self.q = mem_manager.create_tensor_view(index=3, 
                                        length=4)

            if varname == "v":
                
                self.v = mem_manager.create_tensor_view(index=7, 
                                        length=3)

            if varname == "omega":
                
                self.omega = mem_manager.create_tensor_view(index=10, 
                                        length=3)

    class JntState:

        def __init__(self, 
                    n_dofs: int, 
                    mem_manager: SharedMemClient):

            self.q = None # joint positions
            self.v = None # joint velocities

            self.n_dofs = n_dofs

            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")

        def assign_views(self, 
            mem_manager: SharedMemClient,
            varname: str):
            
            if varname == "q":
                
                self.q = mem_manager.create_tensor_view(index=13, 
                                        length=self.n_dofs)
                
            if varname == "v":
                
                self.v = mem_manager.create_tensor_view(index=13 + self.n_dofs, 
                                        length=self.n_dofs)
                
    def __init__(self, 
                n_dofs: int, 
                cluster_size: int,
                index: int,
                dtype = torch.float32, 
                verbose=False):

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self.n_dofs = n_dofs

        # root p, q, v, omega + jnt q, v respectively
        aggregate_view_columnsize = 3 + \
            4 + \
            3 + \
            3 + \
            2 * self.n_dofs # hardcoded
        
        # this creates the view of the shared data for the robot specificed by index
        self.shared_memman = SharedMemClient(cluster_size, 
                        aggregate_view_columnsize, 
                        index, 
                        'RobotState', 
                        self.dtype, 
                        verbose=verbose) # this blocks untils the server creates the associated memory
        
        self.root_state = RobotState.RootState(self.shared_memman) # created as a view of the
        # shared memory pointed to by the manager

        self.jnt_state = RobotState.JntState(n_dofs, 
                                        self.shared_memman) # created as a view of the
        # shared memory pointed to by the manager
        
        # we now make all the data in root_state and jnt_state a view of the memory viewed by the manager
        # paying attention to read the right blocks

class RobotCmds:

    class JntCmd:

        def __init__(self, 
                    n_dofs: int, 
                    dtype = torch.float32, 
                    device: torch.device = torch.device('cpu')):

            self.q = torch.zeros((1, n_dofs), dtype=dtype, device = device) # joint positions
            self.v = torch.zeros((1, n_dofs), dtype=dtype, device = device) # joint velocities
            self.eff = torch.zeros((1, n_dofs), dtype=dtype, device = device) # joint efforts

    class SolverState:

        def __init__(self, 
                add_info_size = 1, 
                dtype = torch.float32, 
                device: torch.device = torch.device('cpu')):

            self.info = torch.zeros((1, add_info_size), dtype=dtype, device = device)

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int,
                index: int,
                add_info_size: int = None, 
                dtype = torch.float32, 
                verbose=False):

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self.n_dofs = n_dofs

        self.add_info_size = add_info_size

        # root p, q, v, omega + jnt q, v respectively
        aggregate_view_columnsize = \
            3 * self.n_dofs + \
            
            # hardcoded
        
        self.jnt_cmd = RobotCmds.JntCmd(n_dofs = n_dofs, 
                                        dtype=self.dtype, 
                                        device=self.device)
        
        if add_info_size is not None:

            self.slvr_state = RobotCmds.SolverState(dtype=self.dtype, 
                                            device=self.device)
            
            self.aggregate_view = torch.cat([
                self.jnt_cmd.q,
                self.jnt_cmd.v,
                self.jnt_cmd.eff,
                self.slvr_state.info
                ], dim=1)
        
        else:

            self.aggregate_view = torch.cat([
                self.jnt_cmd.q,
                self.jnt_cmd.v,
                self.jnt_cmd.eff
                ], dim=1)
            
        self.shared_memman = SharedMemClient(cluster_size, 
                        self.n_dofs, 
                        index, 
                        'RobotCmds', 
                        self.dtype, 
                        verbose=verbose) # this blocks untils the server creates the associated memory
        
RobotStateChild = TypeVar('RobotStateChild', bound='RobotState')
