import torch

from typing import TypeVar, List

from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_utils.utilities.defs import states_name, cmds_name
class RobotState:

    class RootState:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    q_remapping: List[int]):
            
            self.p = None # floating base position
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base angular vel

            self.q_remapping = None
            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)
                
            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "p")
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "omega")

        def __del__(self):

            self.terminate()

        def terminate(self):

            # release any memory view

            self.p = None
            self.q = None
            self.v = None
            self.omega = None

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "p":
                
                self.p = mem_manager.create_partial_tensor_view(index=0, 
                                        length=3)

            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=3, 
                                        length=4)

            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=7, 
                                        length=3)

            if varname == "omega":
                
                self.omega = mem_manager.create_partial_tensor_view(index=10, 
                                        length=3)

        def get_p(self):
            
            return self.p[:, :]
        
        def get_q(self):
            
            if self.q_remapping is not None:

                return self.q[:, self.q_remapping]
            
            else:

                return self.q[:, :]
        
        def get_v(self):
            
            return self.v[:, :]
        
        def get_omega(self):

            return self.omega[:, :]
        
    class JntState:

        def __init__(self, 
                    n_dofs: int, 
                    mem_manager: SharedMemClient, 
                    jnt_remapping: List[int]):

            self.q = None # joint positions
            self.v = None # joint velocities

            self.n_dofs = n_dofs

            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")

            self.jnt_remapping = None
            if self.jnt_remapping is not None: 
                self.jnt_remapping = torch.tensor(jnt_remapping)

        def __del__(self):

            self.terminate()

        def terminate(self):

            # we release any memory view

            self.q = None
            self.v = None

        def assign_views(self, 
            mem_manager: SharedMemClient,
            varname: str):
            
            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=13, 
                                        length=self.n_dofs)
                
            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=13 + self.n_dofs, 
                                        length=self.n_dofs)
        
        def get_q(self):
            
            if self.jnt_remapping is not None:

                return self.q[:, self.jnt_remapping]
            
            else:

                return self.q[:, :]
        
        def get_v(self):
            
            if self.jnt_remapping is not None:

                return self.v[self.jnt_remapping]

            else:

                return self.v[:, :]
        
    def __init__(self, 
                n_dofs: int, 
                cluster_size: int,
                index: int,
                jnt_remapping: List[int],
                q_remapping: List[int],
                dtype = torch.float32, 
                verbose=False):

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self.n_dofs = n_dofs
        self.cluster_size = cluster_size
        aggregate_view_columnsize = aggregate_state_size(self.n_dofs)
        
        self.jnt_remapping = jnt_remapping
        self.q_remapping = q_remapping

        # this creates the view of the shared data for the robot specificed by index
        self.shared_memman = SharedMemClient(n_rows=self.cluster_size, 
                        n_cols=aggregate_view_columnsize, 
                        client_index=index, 
                        name=states_name(), 
                        dtype=self.dtype, 
                        verbose=verbose) 
        self.shared_memman.attach() # this blocks untils the server creates the associated memory

        self.root_state = RobotState.RootState(self.shared_memman, 
                                    self.q_remapping) # created as a view of the
        # shared memory pointed to by the manager

        self.jnt_state = RobotState.JntState(n_dofs, 
                                        self.shared_memman, 
                                        self.jnt_remapping) # created as a view of the
        # shared memory pointed to by the manager
        
        # we now make all the data in root_state and jnt_state a view of the memory viewed by the manager
        # paying attention to read the right blocks

    def __del__(self):

        self.terminate()

    def terminate(self):

        self.root_state.terminate()
        self.jnt_state.terminate()

        self.shared_memman.terminate()

class RobotCmds:

    class JntCmd:

        def __init__(self, 
                    n_dofs: int, 
                    mem_manager: SharedMemClient, 
                    jnt_remapping: List[int]):

            self.n_dofs = n_dofs

            self.q = None # joint positions
            self.v = None # joint velocities
            self.eff = None # joint efforts

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "eff")

            self.jnt_remapping = torch.tensor(jnt_remapping)

        def __del__(self):

            self.terminate()

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=0, 
                                        length=self.n_dofs)
                
            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=self.n_dofs, 
                                        length=self.n_dofs)
                
            if varname == "eff":
                
                self.eff = mem_manager.create_partial_tensor_view(index=2 * self.n_dofs, 
                                        length=self.n_dofs)

        def set_q(self, 
                q: torch.Tensor):
            
            if self.jnt_remapping is not None:
                                
                self.q[:, :] = q[:, self.jnt_remapping]

            else:

                self.q[:, :] = q

        def set_v(self, 
                v: torch.Tensor):
            
            if self.jnt_remapping is not None:

                self.v[:, :] = v[:, self.jnt_remapping]

            else:

                self.v[:, :] = v

        def set_eff(self, 
                eff: torch.Tensor):
            
            if self.jnt_remapping is not None:

                self.eff[:, :] = eff[:, self.jnt_remapping]

            else:

                self.eff[:, :] = eff

        def terminate(self):
            
            # we release all memory views

            self.q = None
            self.v = None
            self.eff = None

    class SolverState:

        def __init__(self, 
                add_info_size: int, 
                n_dofs: int,
                mem_manager: SharedMemClient):

            self.info = None

            self.add_info_size = add_info_size
            
            self.n_dofs = n_dofs

            self.assign_views(mem_manager, "info")

        def __del__(self):

            self.terminate()

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "info":
                
                self.info = mem_manager.create_partial_tensor_view(index= 3 * self.n_dofs, 
                                        length=self.add_info_size)
        
        def set_info(self, 
                info: torch.Tensor):

            self.info[:, :] = info

        def terminate(self):
            
            # we release any memory view
            self.info = None

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int,
                index: int,
                jnt_remapping: List[int],
                add_info_size: int = None, 
                dtype = torch.float32, 
                verbose=False):

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self.n_dofs = n_dofs
        self.cluster_size = cluster_size
        self.add_info_size = add_info_size

        self.jnt_remapping = jnt_remapping

        aggregate_view_columnsize = -1

        if add_info_size is not None:

            aggregate_view_columnsize = aggregate_cmd_size(self.n_dofs, 
                                                self.add_info_size)
        else:

            aggregate_view_columnsize = aggregate_cmd_size(self.n_dofs, 
                                                        0)
        
        self.shared_memman = SharedMemClient(n_rows=self.cluster_size, 
                        n_cols=aggregate_view_columnsize, 
                        client_index=index, 
                        name=cmds_name(), 
                        dtype=self.dtype, 
                        verbose=verbose) # this blocks untils the server creates the associated memory
        self.shared_memman.attach()
        
        self.jnt_cmd = RobotCmds.JntCmd(n_dofs=self.n_dofs, 
                                        mem_manager=self.shared_memman, 
                                        jnt_remapping=self.jnt_remapping)
        
        if add_info_size is not None:

            self.slvr_state = RobotCmds.SolverState(self.add_info_size, 
                                            self.n_dofs, 
                                            self.shared_memman)
    
    def __del__(self):

        self.terminate()

    def terminate(self):

        self.jnt_cmd.terminate()

        self.slvr_state.terminate()

        self.shared_memman.terminate()

RobotStateChild = TypeVar('RobotStateChild', bound='RobotState')
RobotCmdsChild = TypeVar('RobotCmdsChild', bound='RobotCmds')
