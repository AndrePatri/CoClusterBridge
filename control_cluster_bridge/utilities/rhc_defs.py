# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
import torch

from typing import TypeVar, List

from control_cluster_bridge.utilities.shared_mem import SharedMemClient
from control_cluster_bridge.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_bridge.utilities.defs import states_name, cmds_name
from control_cluster_bridge.utilities.defs import aggregate_refs_size, task_refs_name
from control_cluster_bridge.utilities.defs import Journal

from abc import ABC, abstractmethod

class RobotState:

    class RootState:

        def __init__(self, 
                mem_manager: SharedMemClient, 
                q_remapping: List[int] = None,
                prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1 

            self.p = None # floating base position
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base angular vel

            self._terminated = False

            self.q_remapping = None
            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)
            
            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "p")
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "omega")

            self.last_index = self.offset

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
                
                self.p = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)

                self.offset = self.offset + 3

            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=4)

                self.offset = self.offset + 4

            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)

                self.offset = self.offset + 3

            if varname == "omega":
                
                self.omega = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)
                
                self.offset = self.offset + 3

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
                jnt_remapping: List[int],
                prev_index: int = 0):

            self.prev_index = prev_index
            self.last_index = -1 

            self.q = None # joint positions
            self.v = None # joint velocities

            self.n_dofs = n_dofs

            self._terminated = False 

            self.offset = self.prev_index

            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            
            self.last_index = self.offset

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
                
                self.q = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_dofs)
                
                self.offset = self.offset + self.n_dofs

            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_dofs)

                self.offset = self.offset + self.n_dofs

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
                                    self.q_remapping, 
                                    prev_index=0) # created as a view of the
        # shared memory pointed to by the manager

        self.jnt_state = self.JntState(n_dofs, 
                                self.shared_memman, 
                                self.jnt_remapping, 
                                prev_index=self.root_state.last_index) # created as a view of the
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
                    jnt_remapping: List[int] = None,
                    prev_index: int = 0):

            self.prev_index = prev_index
            self.last_index = -1 

            self.n_dofs = n_dofs

            self.q = None # joint positions
            self.v = None # joint velocities
            self.eff = None # joint efforts

            self._terminated = False

            self.jnt_remapping = None

            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "q")
            self.assign_views(mem_manager, "v")
            self.assign_views(mem_manager, "eff")

            self.last_index = self.offset

            if jnt_remapping is not None:

                self.jnt_remapping = torch.tensor(jnt_remapping)

        def __del__(self):

            self.terminate()

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            # we create views 

            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_dofs)
                
                self.offset = self.offset + self.n_dofs

            if varname == "v":
                
                self.v = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_dofs)
                
                self.offset = self.offset + self.n_dofs

            if varname == "eff":
                
                self.eff = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_dofs)

                self.offset = self.offset + self.n_dofs

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
                mem_manager: SharedMemClient,
                prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1 

            self.info = None

            self.add_info_size = add_info_size
            
            self.n_dofs = n_dofs

            self._terminated = False

            self.offset = self.prev_index

            self.assign_views(mem_manager, "info")

            self.last_index = self.offset

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
                                        jnt_remapping=self.jnt_remapping, 
                                        prev_index=0)
        
        if add_info_size is not None:

            self.slvr_state = self.SolverState(self.add_info_size, 
                                            self.n_dofs, 
                                            self.shared_memman, 
                                            prev_index=self.jnt_cmd.last_index)
    
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

    class Phase:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    n_contacts: int,
                    dtype = torch.float32, 
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.phase_id = None # type of current phase (-1 custom, ...)
            self.is_contact = None # array of contact flags for each contact
            self.duration = None # phase duration
            self.p0 = None # start position
            self.p1 = None # end position
            self.clearance = None # flight clearance
            self.d0 = None # initial derivative
            self.d1 = None # end derivative

            self.dtype = dtype

            self.n_contacts = n_contacts

            self._terminated = False
            
            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "phase_id")
            self.assign_views(mem_manager, "is_contact")
            self.assign_views(mem_manager, "duration")
            self.assign_views(mem_manager, "p0")
            self.assign_views(mem_manager, "p1")
            self.assign_views(mem_manager, "clearance")
            self.assign_views(mem_manager, "d0")
            self.assign_views(mem_manager, "d1")

            self.last_index = self.offset

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
            
            if varname == "phase_id":
                
                self.phase_id = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=1)

                self.phase_id[:, :] = -1.0 # by default we run in custom mode
                
                self.offset = self.offset + 1

            if varname == "is_contact":
                
                self.is_contact = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=self.n_contacts)
                
                self.is_contact[:, :] = torch.full(size=(1, self.n_contacts), 
                                                fill_value=1.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + self.n_contacts

            if varname == "duration":

                self.duration = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=1)
                
                self.duration[:, :] = torch.full(size=(1, 1), 
                                                fill_value=0.5, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 1

            if varname == "p0":

                self.p0 = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)
                
                self.p0[:, :] = torch.full(size=(1, 3), 
                                                fill_value=0.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 3
            
            if varname == "p1":

                self.p1 = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)
                
                self.p1[:, :] = torch.full(size=(1, 3), 
                                                fill_value=0.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 3

            if varname == "clearance":

                self.clearance = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=1)
                
                self.clearance[:, :] = torch.full(size=(1, 1), 
                                                fill_value=0.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 1

            if varname == "d0":

                self.d0 = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=1)
                
                self.d0[:, :] = torch.full(size=(1, 1), 
                                                fill_value=0.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 1

            if varname == "d1":

                self.d1 = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=1)
                
                self.d1[:, :] = torch.full(size=(1, 1), 
                                                fill_value=0.0, 
                                                dtype=self.dtype) # by default contact
                
                self.offset = self.offset + 1

        def get_phase_id(self):
            
            return self.phase_id[:, :].item()
        
        def get_contacts(self):
            
            return self.is_contact[:, :]
        
        def get_duration(self):
            
            return self.duration[:, :]

        def get_p0(self):

            return self.p0
        
        def get_p1(self):

            return self.p1
        
        def get_clearance(self):

            return self.clearance
        
        def get_d0(self):

            return self.d0
        
        def get_d1(self):

            return self.d1

        def get_flight_param(self):

            return torch.cat((self.duration, self.p0, self.p1, \
                            self.clearance, self.d0, self.d1), dim=1)
        
        def set_contacts(self, 
                contacts: torch.Tensor):
                                            
            self.is_contact[:, :] = contacts

        def set_phase_id(self, 
                phase_id: int):
                                            
            self.phase_id[:, :] = phase_id

        def set_duration(self, 
                duration: float):
            
            self.duration[:, :] = duration
        
        def set_p0(self, 
                p0: torch.Tensor):
                                            
            self.p0[:, :] = p0
        
        def set_p1(self, 
                p1: torch.Tensor):
                                            
            self.p1[:, :] = p1
        
        def set_clearance(self, 
                clearance: float):
                                            
            self.clearance[:, :] = clearance
        
        def set_d0(self, 
                d0: float):
                                            
            self.d0[:, :] = d0
        
        def set_d1(self, 
                d1: float):
                                            
            self.d1[:, :] = d1

    class BasePose:

        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    q_remapping: List[int] = None,
                    dtype = torch.float32,
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.dtype = dtype

            self.p = None # base position
            self.q = None # base orientation (quaternion)
            self.pose = None # full pose [p, q]

            self._terminated = False

            self.q_remapping = None

            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)
            
            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "pose")
            self.assign_views(mem_manager, "p")
            self.assign_views(mem_manager, "q")

            self.last_index = self.offset

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

            if varname == "pose":
                
                self.pose = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=7)

                init = torch.full(size=(1, 7), 
                                fill_value=0.0,
                                dtype=self.dtype)
                init[0, 3] = 1.0

                self.pose[:, :] = init

            if varname == "p":
                
                self.p = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)

                self.p[:, :] = torch.full(size=(1, 3), 
                                        fill_value=0.0,
                                        dtype=self.dtype)
                
                self.offset = self.offset + 3
                
            if varname == "q":
                
                self.q = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=4)

                init = torch.full(size=(1, 4), 
                                fill_value=0.0,
                                dtype=self.dtype)
                init[0, 0] = 1.0

                self.q[:, :] = init

                self.offset = self.offset + 4
            
        def get_q(self, 
                use_remapping = False):
            
            if self.q_remapping is not None and use_remapping:

                return self.q[:, self.q_remapping]
            
            else:

                return self.q[:, :]
        
        def set_q(self,
                q: torch.tensor):
            
            self.q[:, :] = q

        def get_p(self):

            return self.p[:, :]
        
        def set_p(self, 
                p: torch.tensor):
            
            self.p[:, :] = p
            
        def get_pose(self, 
                use_remapping = False):

            if self.q_remapping is not None and use_remapping:
                
                pose = self.pose.clone()

                pose[:, 3:] = self.q[:, self.q_remapping]

                return pose
            
            else:

                return self.pose[:, :]
        
        def get_base_xy(self):

            return self.p[0:2]
    
    class CoMPose:
        
        def __init__(self, 
                    mem_manager: SharedMemClient, 
                    q_remapping: List[int] = None,
                    dtype = torch.float32,
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.dtype = dtype
            
            self.q_remapping = None

            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)

            self.com_pos = None # com position
            self.com_q = None # com orientation
            self.com_pose = None # com pose

            self._terminated = False
            
            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "com_pose")
            self.assign_views(mem_manager, "com_pos")
            self.assign_views(mem_manager, "com_q")

            self.last_index = self.offset
            
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
            
            if varname == "com_pose":
                
                self.com_pose = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=7)

            if varname == "com_pos":
                
                self.com_pos = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)

                self.com_pos[:, 2] = 0.4

                self.offset = self.offset + 3

            if varname == "com_q":
                
                self.com_q = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=4)

                self.com_q[:, :] = torch.tensor([[1, 0, 0, 0]], dtype=self.dtype)

                self.offset = self.offset + 4

        def get_com_height(self):
            
            return self.com_pos[:, 2].item()

        def get_com_pos(self):
            
            return self.com_pos[:, :]
        
        def get_com_q(self, 
                        remap = True):
            
            if self.q_remapping is not None and remap:

                return self.com_q[:, self.q_remapping]
            
            else:

                return self.com_q[:, :]
                    
        def get_com_pose(self):
            
            return self.com_pose[:, :]
        
        def set_com_height(self, 
                height: float):

            self.com_pos[:, 2] = height

        def set_com_pos(self, 
                pos: torch.tensor):

            self.com_pos[:, :] = pos

        def set_com_q(self, 
                q: torch.tensor):

            self.com_q[:, :] = q

        def set_com_pose(self, 
                pose: torch.tensor):

            self.com_pose[:, :] = pose

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

        self.phase_id = self.Phase(mem_manager=self.shared_memman, 
                                    n_contacts=self.n_contacts,
                                    dtype=self.dtype, 
                                    prev_index = 0) # created as a view of the
        # shared memory pointed to by the manager

        self.base_pose = self.BasePose(mem_manager=self.shared_memman, 
                                    q_remapping=self.q_remapping, 
                                    dtype=self.dtype, 
                                    prev_index = self.phase_id.last_index)

        self.com_pose = self.CoMPose(mem_manager=self.shared_memman, 
                                q_remapping=self.q_remapping,
                                dtype=self.dtype, 
                                prev_index = self.base_pose.last_index)

    def __del__(self):
        
        if not self._terminated:

            self.terminate()

    def terminate(self):

        if not self._terminated:

            self.phase_id.terminate()

            self.base_pose.terminate()

            self.com_pose.terminate()

            self.shared_memman.terminate()

            self._terminated = True

    @abstractmethod
    def update(self):

        pass

RobotStateChild = TypeVar('RobotStateChild', bound='RobotState')
RobotCmdsChild = TypeVar('RobotCmdsChild', bound='RobotCmds')
RhcTaskRefsChild = TypeVar('RhcTaskRefsChild', bound='RhcTaskRefs')
