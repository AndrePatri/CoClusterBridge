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

from control_cluster_bridge.utilities.shared_mem import SharedMemClient, SharedStringArray
from control_cluster_bridge.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_bridge.utilities.defs import states_name, cmds_name
from control_cluster_bridge.utilities.defs import contacts_info_name,contacts_names
from control_cluster_bridge.utilities.defs import aggregate_refs_size, task_refs_name
from control_cluster_bridge.utilities.defs import Journal

from abc import ABC, abstractmethod

import numpy as np

from control_cluster_bridge.utilities.shared_mem import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal as Logger
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype, toNumpyDType

from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

class ContactState:

    class ContactState:

        def __init__(self, 
                n_contacts: int,
                contact_names: List[str],
                mem_manager: SharedMemClient, 
                prev_index: int = 0):
            
            self.journal = Journal()
            
            self.prev_index = prev_index
            self.last_index = -1 

            self.n_contacts = n_contacts
            self.contact_names = contact_names

            self.net_contact_forces = [None] * self.n_contacts 

            self._terminated = False
            
            self.offset = self.prev_index

            # we assign the right view of the raw shared data
            self.assign_views(mem_manager, "p")
            
            self.last_index = self.offset

        def __del__(self):

            self.terminate()

        def terminate(self):
            
            if not self._terminated:
                
                self._terminated = True

                # release any memory view

                self.net_contact_forces = None

        def assign_views(self, 
                    mem_manager: SharedMemClient,
                    varname: str):
            
            for i in range(0, self.n_contacts):

                self.net_contact_forces[i] = mem_manager.create_partial_tensor_view(index=self.offset, 
                                        length=3)

                self.offset = self.offset + 3

        def get(self, 
                contact_name: str):

            index = -1

            if contact_name == "":

                exception = f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                    f"An empty contact name was provided!" 
            
                raise Exception(exception)
            
            try:
            
                index = self.contact_names.index(contact_name)
        
            except:
                
                exception = f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                    f"could not find contact link {contact_name} in contact list {' '.join(self.contact_names)}." 
            
                raise Exception(exception)
            
            return self.net_contact_forces[index]

    def __init__(self, 
                index: int,
                dtype = torch.float32,
                namespace = "", 
                verbose=False):

        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.device = torch.device('cpu') # born to live on CPU

        self._terminated = False

        # contact names
        self.contact_names_shared = SharedStringArray(length=-1, 
                                    name=contacts_names(), 
                                    namespace=self.namespace,
                                    is_server=False, 
                                    wait_amount=0.1, 
                                    verbose=verbose)
        self.contact_names_shared.start()

        self.contact_names  = self.contact_names_shared.read()

        self.n_contacts = len(self.contact_names)

        # this creates the view of the shared data for the robot specificed by index
        self.shared_memman = SharedMemClient(
                        client_index=index, 
                        name=contacts_info_name(), 
                        namespace=self.namespace,
                        dtype=self.dtype, 
                        verbose=verbose) 
        self.shared_memman.attach() # this blocks untils the server creates the associated memory

        self.contact_state = self.ContactState(self.n_contacts, 
                                self.contact_names,
                                self.shared_memman, 
                                prev_index=0) 

    def __del__(self):
        
        if not self._terminated:
        
            self.terminate()

    def terminate(self):

        if not self._terminated:

            self.contact_names_shared.terminate()

            self.shared_memman.terminate()

            self.contact_names_shared.terminate()

            self._terminated = True

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
            if jnt_remapping is not None: 
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

                return self.v[:, self.jnt_remapping]

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

                self.com_pos[:, 2] = 0.5

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

class RHCDebugData():

    # shared aggregate data object.
    # useful for sharing and debugging rhc data like
    # costs and constraints 

    class Names():
        
        def __init__(self,
            names: List[str] = None,
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0):

            basename = "debug_data_names"
            
            self.is_server = is_server

            self.names = names

            if self.is_server:

                self.shared_names = StringTensorServer(length = len(names), 
                                            basename = basename, 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel, 
                                            force_reconnection = True)

            else:

                self.shared_names = StringTensorClient(
                                            basename = basename, 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel)
                
        def run(self):
            
            self.shared_names.run()
            
            if self.is_server:
                
                jnt_names_written = self.shared_names.write_vec(self.names, 0)

                if not jnt_names_written:

                    raise Exception("Could not write joint names on shared memory!")
            
            else:
                
                self.names = [""] * self.shared_names.length()

                jnt_names_read = self.shared_names.read_vec(self.names, 0)

                if not jnt_names_read:

                    raise Exception("Could not read joint names on shared memory!")

        def close(self):

            self.shared_names.close()
            
    class DataDims(SharedDataView):
        
        def __init__(self,
            dims: List[int] = None,
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0):
        
            basename = "debug_data_dims"

            self.dims = dims
            
            n_dims = None

            if is_server:

                n_dims = 0
                for i in range(0, len(dims)):

                    n_dims += dims[i]
                
                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    n_rows = n_dims, 
                    n_cols = 1, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    dtype=sharsor_dtype.Int)

            else:

                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    dtype=sharsor_dtype.Int)
                
        def run(self):

            super().run()

            if self.is_server:

                dims = np.array(self.dims, dtype=toNumpyDType(self.shared_mem.getScalarType())).reshape((len(self.dims), 1))

                self.write_wait(data = dims, row_index= 0,
                        col_index=0)
            
            else:

                # updates shared dims
                self.synch_all(read=True, wait=True)
                
                self.dims = self.numpy_view[:, :].copy()

                self.n_dims = self.n_rows
                self.n_nodes = self.n_cols
                
    class Data(SharedDataView):
        
        def __init__(self,
            namespace = "",
            is_server = False, 
            n_dims: int = -1, 
            n_nodes: int = -1, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0):
        
            basename = "debug_data" 

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value=np.nan)

    def __init__(self,
            names: List[str] = None, # not needed if client
            dimensions: List[int] = None, # not needed if client
            n_nodes: int = -1, # not needed if client 
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0):
        
        self.names = names
        self.dimensions = dimensions

        self.n_dims = None
        self.n_nodes = n_nodes

        self.is_server = is_server

        if self.is_server:
            
            n_dims = 0

            for i in range(0, len(dimensions)):

                n_dims = n_dims + dimensions[i]

            self.n_dims = n_dims

        # actual data
        self.data = self.Data(namespace = namespace,
                is_server = is_server, 
                n_dims= self.n_dims, 
                n_nodes = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
        
        # names of each block of data
        self.shared_names = self.Names(namespace = namespace,
                is_server = is_server, 
                names = self.names,
                verbose = verbose, 
                vlevel = vlevel)

        # dimenions of each block of data
        self.shared_dims = self.DataDims(namespace = namespace,
                is_server = is_server, 
                dims = dimensions,
                verbose = verbose, 
                vlevel = vlevel)

    def run(self):
        
        self.data.run()

        self.shared_names.run()

        self.shared_dims.run()

        if not self.is_server:

            self.names = self.shared_names.names
            
            # updates shared dims
            self.shared_dims.synch_all(read=True, wait=True)

            self.dimensions = self.shared_dims.dims.flatten().tolist()

            self.n_dims = self.data.n_rows
            self.n_nodes = self.data.n_cols

    def write(self,
        data: np.ndarray,
        name: str,
        wait = True):

        # we assume names does not contain
        # duplicates
        
        data_idx = self.names.index(name)

        # we sum dimensions up until the data we 
        # need to write to get the starting index
        # of the data block
        starting_idx = 0
        for index in range(data_idx):

            starting_idx += self.dimensions[index]

        data_2D = np.atleast_2d(data)

        if wait: 

            self.data.write_wait(np.atleast_2d(data_2D), starting_idx, 0) # blocking
            
            return True
        
        else:

            return self.data.write(np.atleast_2d(data_2D), starting_idx, 0) # non-blocking

    def synch(self,
            wait = True):

        # to be called before using get() on one or more data 
        # blocks

        # updates the whole view with shared data
        return self.data.synch_all(read = True, wait = wait)

    def get(self,
        name: str):

        # we assume names does not contain
        # duplicates
        data_idx = self.names.index(name)

        # we sum dimensions up until the data we 
        # need to write to get the starting index
        # of the data block
        starting_idx = - 1
        for index in range(data_idx + 1):

            starting_idx += self.dimensions[index]

        view = self.data.numpy_view[starting_idx:starting_idx + self.dimensions[index], :]

        view_copy = view.copy()

        return view_copy
    
    def close(self):

        self.data.close()

        self.shared_names.close()

        self.shared_dims.close()
            
class RHCInternal():

    # shared data for an acceleration-based
    # formulation RHC controller

    class Q(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "q" # configuration vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class V(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "q" # velocity vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class A(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "A" # acceleration vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class ADot(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "a_dot" # jerk vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class F(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "f" # cartesian force vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
            
    class FDot(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "f_dot" # yank vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)
            
    class Eff(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "v" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel)

    class RHCosts(RHCDebugData):

        def __init__(self,
                names: List[str] = None, # not needed if client
                dimensions: List[int] = None, # not needed if client
                n_nodes: int = -1, # not needed if client 
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "rhc_costs"

            super().__init__(names = names, # not needed if client
                    dimensions = dimensions, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = namespace + basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel) 
    
    class RHConstr(RHCDebugData):

        def __init__(self,
                names: List[str] = None, # not needed if client
                dimensions: List[int] = None, # not needed if client
                n_nodes: int = -1, # not needed if client 
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "rhc_constraints"

            super().__init__(names = names, # not needed if client
                    dimensions = dimensions, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = namespace + basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel) 
    
    class Config():

        def __init__(self,
            is_server: bool = False,
            enable_q: bool = True, 
            enable_v: bool = True, 
            enable_a: bool = True,
            enable_a_dot: bool = False, 
            enable_f: bool = True, 
            enable_f_dot: bool = False, 
            enable_eff: bool = True, 
            cost_names: List[str] = None, 
            constr_names: List[str] = None,
            cost_dims: List[int] = None, 
            constr_dims: List[int] = None,
            enable_costs: bool = False,
            enable_constr: bool = False):

            self.is_server = is_server

            self.enable_q = enable_q
            self.enable_v = enable_v
            self.enable_a = enable_a
            self.enable_a_dot = enable_a_dot
            self.enable_f = enable_f
            self.enable_f_dot = enable_f_dot
            self.enable_eff = enable_eff

            self.enable_costs = enable_costs
            self.enable_constr = enable_constr

            self.cost_names = None
            self.cost_dims = None

            self.constr_names = None
            self.constr_dims = None

            self.n_costs = 0
            self.n_constr = 0

            self._set_cost_data(cost_names, cost_dims)
            self._set_constr_data(constr_names, constr_dims)


        def _set_cost_data(self, 
                        names: List[str] = None,
                        dims: List[str] = None):

            self.cost_names = names
            self.cost_dims = dims

            if (names is not None) and (self.cost_names is None) and \
                self.is_server:
                
                excep = "Cost enabled but no cost_names list was provided"

                raise Exception(excep)

            if (dims is not None) and (self.cost_dims is None) and \
                self.is_server:
                
                excep = "Cost enabled but no cost_dims list was provided"

                raise Exception(excep)

            if self.is_server and (not (len(self.cost_names) == len(self.cost_dims))):
                
                excep = f"Cost names dimension {len(self.cost_names)} " + \
                    f"does not match dim. vector length {len(self.cost_dims)}"

                raise Exception(excep)
            
            if self.is_server:

                self.enable_costs = True
                self.n_costs = len(self.cost_names)

        def _set_constr_data(self, 
                        names: List[str] = None,
                        dims: List[str] = None):

            self.constr_names = names
            self.constr_dims = dims

            if (names is not None) and (self.constr_names is None) and \
                self.is_server:
                
                excep = "Constraints enabled but no cost_names list was provided"

                raise Exception(excep)

            if (dims is not None) and (self.constr_dims is None) and \
                self.is_server:
                
                excep = "Cost enabled but no constr_dims list was provided"

                raise Exception(excep)

            if self.is_server and (not (len(self.constr_names) == len(self.constr_dims))):

                excep = f"Cost names dimension {len(self.constr_names)} " + \
                    f"does not match dim. vector length {len(self.constr_dims)}"

                raise Exception(excep)
            
            if self.is_server:

                self.enable_constr = True
                self.n_constr = len(self.constr_names)

    def __init__(self,
            config: Config = None,
            namespace = "",
            rhc_index = 0,
            is_server = False, 
            n_nodes: int = -1, 
            n_contacts: int = -1,
            n_jnts: int = -1,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0):

        self.rhc_index = rhc_index

        # appending controller index to namespace
        self.namespace = namespace + "_n_" + str(self.rhc_index)

        if config is not None:

            self.config = config
        
        else:
            
            # use defaults
            self.config = self.Config()

        self.finalized = False

        self.q = None
        self.v = None
        self.a = None
        self.a_dot = None
        self.f = None
        self.f_dot = None
        self.eff = None
        self.costs = None
        self.cnstr = None
            
        if self.config.enable_q:
            
            self.q = self.Q(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 4 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_v:

            self.v = self.V(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_a:

            self.a = self.A(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_a_dot:

            self.a_dot = self.ADot(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_f:

            self.f = self.F(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
            
        if self.config.enable_f_dot:

            self.f_dot = self.FDot(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_eff:

            self.eff = self.Eff(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel)
            
        if self.config.enable_costs:

            self.costs = self.RHCosts(names = self.config.cost_names, # not needed if client
                    dimensions = self.config.cost_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel)
        
        if self.config.enable_constr:

            self.cnstr = self.RHConstr(names = self.config.constr_names, # not needed if client
                    dimensions = self.config.constr_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel)
    
    def run(self):

        if self.q is not None:

            self.q.run()
        
        if self.v is not None:

            self.v.run()
        
        if self.a is not None:

            self.a.run()
        
        if self.a_dot is not None:

            self.a_dot.run()
        
        if self.f is not None:

            self.f.run()
            
        if self.f_dot is not None:

            self.f_dot.run()
        
        if self.eff is not None:

            self.eff.run()
            
        if self.costs is not None:

            self.costs.run()
        
        if self.cnstr is not None:

            self.cnstr.run()

        self.finalized = True

    def synch(self):
        
        # to be used to read updated data 
        # (before calling any read method)
        # it synchs all available data
        
        if self.q is not None:

            self.q.synch()
        
        if self.v is not None:

            self.v.synch()
        
        if self.a is not None:

            self.a.synch()
        
        if self.a_dot is not None:

            self.a_dot.synch()
        
        if self.f is not None:

            self.f.synch()
            
        if self.f_dot is not None:

            self.f_dot.synch()
        
        if self.eff is not None:

            self.eff.synch()
            
        if self.costs is not None:

            self.costs.synch()
        
        if self.cnstr is not None:

            self.cnstr.synch()

    def close(self):

        if self.q is not None:

            self.q.close()
        
        if self.v is not None:

            self.v.close()
        
        if self.a is not None:

            self.a.close()
        
        if self.a_dot is not None:

            self.a_dot.close()
        
        if self.f is not None:

            self.f.close()
            
        if self.f_dot is not None:

            self.f_dot.close()
        
        if self.eff is not None:

            self.eff.close()
            
        if self.costs is not None:

            self.costs.close()
        
        if self.cnstr is not None:

            self.cnstr.close()

    def write_q(self, 
                data: np.ndarray = None,
                wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.q is not None) and (data is not None):
            
            if wait:
                
                self.q.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.q.write(data=data,
                        row_index=0, col_index=0)
    
    def write_v(self, 
            data: np.ndarray = None,
            wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.v is not None) and (data is not None):
            
            if wait:
                
                self.v.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.v.write(data=data,
                        row_index=0, col_index=0)

    def write_a(self, 
            data: np.ndarray = None,
            wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.a is not None) and (data is not None):
            
            if wait:
                
                self.a.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.a.write(data=data,
                        row_index=0, col_index=0)
            
    def write_a_dot(self, 
        data: np.ndarray = None,
        wait = True):

        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.a_dot is not None) and (data is not None):
            
            if wait:
                
                self.a_dot.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.a_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f(self, 
        data: np.ndarray = None,
        wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.f is not None) and (data is not None):
            
            if wait:
                
                self.f.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.f.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f_dot(self, 
        data: np.ndarray = None,
        wait = True):

        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.f is not None) and (data is not None):
            
            if wait:
                
                self.f_dot.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.f_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_eff(self, 
        data: np.ndarray = None,
        wait = True):

        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")

        if (self.eff is not None) and (data is not None):
            
            if wait:
                
                self.eff.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.eff.write(data=data,
                        row_index=0, col_index=0)
                
    def write_cost(self, 
                cost_name: str,
                data: np.ndarray = None,
                wait = True):

        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.costs is not None) and (data is not None):

            self.costs.write(data = data, 
                            name=cost_name,
                            wait=wait)
    
    def read_cost(self, 
            cost_name: str,
            wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")

        if self.costs is not None:
            
            return self.costs.get(cost_name)
        
        else:
            
            raise Exception("Cannot retrieve costs. Make sure to provide cost names and dims to Config.")
            
    def write_constr(self, 
                constr_name: str,
                data: np.ndarray = None,
                wait = True):

        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if (self.cnstr is not None) and (data is not None):
            
            self.cnstr.write(data = data, 
                            name=constr_name,
                            wait=wait)
            
    def read_constr(self, 
            constr_name,
            wait = True):
        
        if not self.finalized:

            raise Exception("RHCInternal not initialized. Did you call the run()?")
        
        if self.cnstr is not None:
            
            return self.cnstr.get(constr_name)
        
        else:
            
            raise Exception("Cannot retrieve constraint. Make sure to provide constraint names and dims to Config.")