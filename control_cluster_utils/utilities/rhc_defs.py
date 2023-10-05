import torch

from typing import TypeVar, List

from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_utils.utilities.defs import states_name, cmds_name
from control_cluster_utils.utilities.defs import aggregate_refs_size, task_refs_name
from control_cluster_utils.utilities.defs import Journal

from abc import ABC, abstractmethod

class RobotState:

    class RootState:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    q_remapping: List[int] = None):
            
            self.p = None # floating base position
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base angular vel

            self._terminated = False

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
            
            if not self._terminated:
                
                self._terminated = True

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

            self._terminated = False 

            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")

            self.jnt_remapping = None
            if self.jnt_remapping is not None: 
                self.jnt_remapping = torch.tensor(jnt_remapping)

        def __del__(self):

            self.terminate()

        def terminate(self):

            if not self._terminated:
                
                self._terminated = True

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
                index: int,
                jnt_remapping: List[int] = None,
                q_remapping: List[int] = None,
                dtype = torch.float32,
                namespace = "", 
                verbose=False):

        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self._terminated = False

        self.jnt_remapping = jnt_remapping
        self.q_remapping = q_remapping

        # this creates the view of the shared data for the robot specificed by index
        self.shared_memman = SharedMemClient(
                        client_index=index, 
                        name=states_name(), 
                        namespace=self.namespace,
                        dtype=self.dtype, 
                        verbose=verbose) 
        self.shared_memman.attach() # this blocks untils the server creates the associated memory

        self.root_state = self.RootState(self.shared_memman, 
                                    self.q_remapping) # created as a view of the
        # shared memory pointed to by the manager

        self.jnt_state = self.JntState(n_dofs, 
                                self.shared_memman, 
                                self.jnt_remapping) # created as a view of the
        # shared memory pointed to by the manager
        
        # we now make all the data in root_state and jnt_state a view of the memory viewed by the manager
        # paying attention to read the right blocks

    def __del__(self):
        
        if not self._terminated:
        
            self.terminate()

    def terminate(self):

        if not self._terminated:

            self.root_state.terminate()

            self.jnt_state.terminate()

            self.shared_memman.terminate()

            self._terminated = True

class RobotCmds:

    class JntCmd:

        def __init__(self, 
                    n_dofs: int, 
                    mem_manager: SharedMemClient, 
                    jnt_remapping: List[int] = None):

            self.n_dofs = n_dofs

            self.q = None # joint positions
            self.v = None # joint velocities
            self.eff = None # joint efforts

            self._terminated = False

            self.jnt_remapping = None

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "eff")

            if jnt_remapping is not None:

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
            
            if not self._terminated:

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

            self._terminated = False

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
            
            if not self._terminated:

                # we release any memory view
                self.info = None

    def __init__(self, 
                n_dofs: int, 
                index: int,
                jnt_remapping: List[int] = None,
                add_info_size: int = None, 
                dtype = torch.float32, 
                namespace = "",
                verbose=False):

        self.journal = Journal()

        self.namespace = namespace
        
        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self.n_dofs = n_dofs
        self.add_info_size = add_info_size

        self.jnt_remapping = jnt_remapping 
        
        self._terminated = False

        self.shared_memman = SharedMemClient(
                        client_index=index, 
                        name=cmds_name(), 
                        namespace=self.namespace,
                        dtype=self.dtype, 
                        verbose=verbose) # this blocks untils the server creates the associated memory
        self.shared_memman.attach()
        
        self.jnt_cmd = self.JntCmd(n_dofs=self.n_dofs, 
                                        mem_manager=self.shared_memman, 
                                        jnt_remapping=self.jnt_remapping)
        
        if add_info_size is not None:

            self.slvr_state = self.SolverState(self.add_info_size, 
                                            self.n_dofs, 
                                            self.shared_memman)
    
    def __del__(self):

        if not self._terminated:
        
            self.terminate()

    def terminate(self):
        
        if not self._terminated:

            self.jnt_cmd.terminate()

            self.slvr_state.terminate()

            self.shared_memman.terminate()

            self._terminated = True
            
class RhcTaskRefs:

    class PhaseId:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    n_contacts: int,
                    dtype = torch.float32):
            
            self.phase_id = None # type of current phase (-1 custom, ...)
            self.is_contact = None # array of contact flags for each contact

            self.dtype = dtype

            self.n_contacts = n_contacts

            self._terminated = False
            
            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "phase_id")
            self.assign_views(mem_manager, "is_contact")

        def __del__(self):

            self.terminate()

        def terminate(self):
            
            if not self._terminated:
                
                self._terminated = True

                # release any memory view

                self.phase_id = None
    
        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "phase_id":
                
                self.phase_id = mem_manager.create_partial_tensor_view(index=0, 
                                        length=1)

                self.phase_id[:, :] = -1.0 # by default we run in custom mode
                
            if varname == "is_contact":
                
                self.is_contact = mem_manager.create_partial_tensor_view(index=1, 
                                        length=self.n_contacts)
                
                self.is_contact[:, :] = torch.full(size=(1, self.n_contacts), 
                                                fill_value=1.0, 
                                                dtype=self.dtype) # by default contact
                
        def get_phase_id(self):
            
            return self.phase_id[:, :].item()
        
        def get_contacts(self):
            
            return self.is_contact[:, :]
        
        def set_contacts(self, 
                contacts: torch.Tensor):
                                            
            self.is_contact[:, :] = contacts

        def set_phase_id(self, 
                phase_id: int):
                                            
            self.phase_id[:, :] = phase_id

    class BasePose:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    n_contacts: int,
                    q_remapping: List[int] = None,
                    dtype = torch.float32):
            
            self.dtype = dtype

            self.p = None # base position
            self.q = None # base orientation (quaternion)
            self.pose = None # full pose [p, q]

            self.n_contacts = n_contacts

            self._terminated = False

            self.q_remapping = None

            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)
                
            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "p")
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "pose")

        def __del__(self):

            self.terminate()

        def terminate(self):
            
            if not self._terminated:
                
                self._terminated = True

                # release any memory view

                self.p = None
                self.q = None
                self.pose = None

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "p":
                
                self.p = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts, 
                                        length=3)

                self.p[:, :] = torch.full(size=(1, 3), 
                                        fill_value=0.0,
                                        dtype=self.dtype)
                
            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts + 3, 
                                        length=4)

                init = torch.full(size=(1, 4), 
                                fill_value=0.0,
                                dtype=self.dtype)
                init[0, 0] = 1.0

                self.q[:, :] = init

            if varname == "pose":
                
                self.pose = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts, 
                                        length=7)

                init = torch.full(size=(1, 7), 
                                fill_value=0.0,
                                dtype=self.dtype)
                init[0, 3] = 1.0

                self.pose[:, :] = init

        def get_q(self):
            
            if self.q_remapping is not None:

                return self.q[:, self.q_remapping]
            
            else:

                return self.q[:, :]
        
        def get_p(self):

            return self.p[:, :]
        
        def get_pose(self):

            return self.pose[:, :]
        
        def get_base_xy(self):

            return self.p[0:2]
        
    class ComPos:
        
        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    n_contacts: int,
                    q_remapping: List[int] = None,
                    dtype = torch.float32):
            
            self.dtype = dtype

            self.n_contacts = n_contacts
            
            self.q_remapping = None

            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)

            self.com_pos = None # com position
            self.com_q = None # com orientation
            self.com_pose = None # com pose

            self._terminated = False
            
            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "com_pose")
            self.assign_views(mem_manager, "com_pos")
            self.assign_views(mem_manager, "com_q")

        def __del__(self):

            self.terminate()

        def terminate(self):
            
            if not self._terminated:
                
                self._terminated = True

                # release any memory view

                self.com_pos = None
                self.com_q = None 
                self.com_pose = None
    
        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "com_pose":
                
                self.com_pose = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts + 7, 
                                        length=7)

                self.com_pose[:, :] = torch.tensor([0])

            if varname == "com_pos":
                
                self.com_pos = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts + 7, 
                                        length=3)

                self.com_pos[:, 2] = 0.4
                
            if varname == "com_q":
                
                self.com_q = mem_manager.create_partial_tensor_view(index=1 + self.n_contacts + 7 + 3, 
                                        length=4)

        def get_com_height(self):
            
            return self.com_pos[:, 2].item()
        
        def get_com_pos(self):
            
            return self.com_pos[:, :]
        
        def get_com_orientation(self):
            
            if self.q_remapping is not None:

                return self.com_q[:, self.q_remapping]
            
            else:

                return self.com_q[:, :]
                    
        def get_com_pose(self):
            
            return self.com_pose[:, :]
        
    def __init__(self, 
                n_contacts: int,
                index: int,
                q_remapping: List[int] = None,
                dtype = torch.float32, 
                namespace = "",
                verbose=False):

        self.journal = Journal()
        
        self.namespace = namespace

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self._terminated = False

        self.n_contacts = n_contacts
        
        self.q_remapping = q_remapping

        # this creates the view of the shared refs data for the robot specificed by index
        self.shared_memman = SharedMemClient(client_index=index, 
                        name=task_refs_name(), 
                        namespace=self.namespace,
                        dtype=self.dtype, 
                        verbose=verbose) 
        self.shared_memman.attach() # this blocks untils the server creates the associated memory

        self.phase_id = self.PhaseId(mem_manager=self.shared_memman, 
                                    n_contacts=self.n_contacts,
                                    dtype=self.dtype) # created as a view of the
        # shared memory pointed to by the manager

        self.base_pose = self.BasePose(mem_manager=self.shared_memman, 
                                    n_contacts=self.n_contacts,
                                    q_remapping=self.q_remapping, 
                                    dtype=self.dtype)

        self.com_pos = self.ComPos(mem_manager=self.shared_memman, 
                                n_contacts=self.n_contacts,
                                q_remapping=self.q_remapping,
                                dtype=self.dtype)

    def __del__(self):
        
        if not self._terminated:

            self.terminate()

    def terminate(self):

        if not self._terminated:

            self.phase_id.terminate()

            self.base_pose.terminate()

            self.com_pos.terminate()

            self.shared_memman.terminate()

            self._terminated = True

    @abstractmethod
    def update(self):

        pass

RobotStateChild = TypeVar('RobotStateChild', bound='RobotState')
RobotCmdsChild = TypeVar('RobotCmdsChild', bound='RobotCmds')
RhcTaskRefsChild = TypeVar('RhcTaskRefsChild', bound='RhcTaskRefs')
