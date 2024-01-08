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
import numpy as np

import time 

from typing import List, Union

from control_cluster_bridge.utilities.shared_mem import SharedMemSrvr, SharedMemClient, SharedStringArray

from control_cluster_bridge.utilities.shared_mem import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from control_cluster_bridge.utilities.defs import aggregate_cmd_size, aggregate_state_size, aggregate_refs_size
from control_cluster_bridge.utilities.defs import contact_state_size, contacts_info_name
from control_cluster_bridge.utilities.defs import states_name, contacts_names, cmds_name, task_refs_name
from control_cluster_bridge.utilities.defs import cluster_size_name, additional_data_name, n_contacts_name
from control_cluster_bridge.utilities.defs import jnt_number_client_name
from control_cluster_bridge.utilities.defs import jnt_names_client_name
from control_cluster_bridge.utilities.defs import Journal

class RobotClusterContactState:

    class ContactStates:

        def __init__(self, 
            cluster_aggregate: torch.Tensor,
            n_contacts: int,
            contact_names: List[str],
            prev_index: int = 0):
            
            self.journal = Journal()
            
            self.prev_index = prev_index
            self.last_index = -1

            self.n_contacts = n_contacts
            self.contact_names = contact_names

            self.cluster_size = cluster_aggregate.shape[0]
        
            self.net_contact_forces = [None] * self.n_contacts # list of torch views

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate)

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor):
            
            for i in range(0, self.n_contacts):

                self.net_contact_forces[i] = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                            3)
            
                self.offset = self.offset + 3

        def get(self, 
                contact_name: str):
            
            index = -1
            try:
            
                index = self.contact_names.index(contact_name)
        
            except:
                
                exception = f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                    f"could not find contact link {contact_name} in contact list {' '.join(self.contact_names)}." 
            
                raise Exception(exception)
            
            return self.net_contact_forces[index]
        
    def __init__(self, 
                cluster_size: int,
                n_contacts: int, 
                contact_names: List[str] = None,
                namespace = "",
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"), 
                dtype: torch.dtype = torch.float32, 
                verbose = True):
        
        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.backend = "torch" # forcing torch backend
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_contacts = n_contacts
        self.contact_names = contact_names

        if self.contact_names is not None:

            if len(self.contact_names) != self.n_contacts:
                
                exception = f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                    f": provided contact names length {len(self.contact_names)} "+ \
                    "does not match {self.n_contacts}"

                raise Exception(exception)
        else:

            self.contact_names = ["no_name"] * self.n_contacts

        self.contact_names_shared = SharedStringArray(length=self.n_contacts, 
                                    name=contacts_names(), 
                                    namespace=self.namespace,
                                    is_server=True, 
                                    wait_amount=0.1, 
                                    verbose=verbose)
        
        cluster_aggregate_columnsize = contact_state_size(self.n_contacts)

        self._terminate = False

        self.cluster_aggregate = torch.zeros(
                                    (self.cluster_size, 
                                        cluster_aggregate_columnsize 
                                    ), 
                                    dtype=self.dtype, 
                                    device=self.device)

        # views of cluster_aggregate
        self.contact_state = self.ContactStates(self.cluster_aggregate, 
                                    self.n_contacts, 
                                    self.contact_names,
                                    prev_index=0)

        # this creates a shared memory block of the right size for the state
        # and a corresponding view of it
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=contacts_info_name(),
                                    namespace=self.namespace, 
                                    dtype=self.dtype)
    
    def start(self):
        
        self.contact_names_shared.start(init = self.contact_names)

        self.shared_memman.start() # will actually initialize the server

    def synch(self):

        # copies contact state from GPU to CPU if necessary
        
        if self.cluster_aggregate.device.type == 'cuda':

            self.shared_memman.tensor_view[:, :] = self.cluster_aggregate.cpu()

            torch.cuda.synchronize() # this way we ensure that after this the state on GPU
            # is fully updated

        else:
            
            self.shared_memman.tensor_view[:, :] = self.cluster_aggregate

    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self.shared_memman.terminate()
            
            self.contact_names_shared.terminate()

    def __del__(self):
        
        self.terminate()

class RobotClusterState:

    class RootStates:

        def __init__(self, 
                cluster_aggregate: torch.Tensor,
                prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1 

            self.p = None # floating base positions
            self.q = None # floating base orientation (quaternion)
            self.v = None # floating base linear vel
            self.omega = None # floating base linear vel

            self.cluster_size = cluster_aggregate.shape[0]

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "p")
            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")
            self.assign_views(cluster_aggregate, "omega")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "p":

                # (can only make views of contigous memory)
                
                self.p = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                                    3)
                
                self.offset = self.offset + 3

            if varname == "q":
                
                self.q = cluster_aggregate[:, self.offset:(self.offset + 4)].view(self.cluster_size, 
                                                                    4)

                self.offset = self.offset + 4

            if varname == "v":
                
                self.v = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                                    3)
                
                self.offset = self.offset + 3

            if varname == "omega":
                
                self.omega = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                                    3)
                
                self.offset = self.offset + 3

    class JntStates:

        def __init__(self, 
                    cluster_aggregate: torch.Tensor,
                    n_dofs: int,
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.n_dofs = n_dofs

            self.cluster_size = cluster_aggregate.shape[0]
        
            self.q = None # joint positions
            self.v = None # joint velocities

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "q":

                # (can only make views of contigous memory)
                
                self.q = cluster_aggregate[:, self.offset:(self.offset + self.n_dofs)].view(self.cluster_size, 
                                                self.n_dofs)
                
                self.offset = self.offset + self.n_dofs

            if varname == "v":
                
                self.v = cluster_aggregate[:, self.offset:(self.offset + self.n_dofs)].view(self.cluster_size, 
                                                self.n_dofs)
                
                self.offset = self.offset + self.n_dofs
    
    def __init__(self, 
                n_dofs: int, 
                cluster_size: int,
                namespace = "",
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"), 
                dtype: torch.dtype = torch.float32, 
                verbose = True):
        
        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.backend = "torch" # forcing torch backend
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_dofs = n_dofs

        cluster_aggregate_columnsize = aggregate_state_size(self.n_dofs)

        self._terminate = False

        self.cluster_aggregate = torch.zeros(
                                    (self.cluster_size, 
                                        cluster_aggregate_columnsize 
                                    ), 
                                    dtype=self.dtype, 
                                    device=self.device)

        # views of cluster_aggregate
        self.root_state = self.RootStates(self.cluster_aggregate, 
                                        prev_index=0) 
        self.jnt_state = self.JntStates(self.cluster_aggregate, 
                                    self.n_dofs, 
                                    prev_index=self.root_state.last_index)

        # this creates a shared memory block of the right size for the state
        # and a corresponding view of it
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=states_name(),
                                    namespace=self.namespace, 
                                    dtype=self.dtype)
    
    def start(self):

        self.shared_memman.start() # will actually initialize the server

    def synch(self):

        # synchs root_state and jnt_state (which will normally live on GPU)
        # with the shared state data using the aggregate view (normally on CPU)

        # this requires a COPY FROM GPU TO CPU
        # (better to use the aggregate to exploit parallelization)

        self.shared_memman.tensor_view[:, :] = self.cluster_aggregate.cpu()

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self.shared_memman.terminate()
            
    def __del__(self):
        
        self.terminate()

class RobotClusterCmd:

    class JntCmd:

        def __init__(self,
                    cluster_aggregate: torch.Tensor, 
                    n_dofs: int,
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self._cluster_size = cluster_aggregate.shape[0]

            self._n_dofs = n_dofs

            self.q = None # joint positions
            self.v = None # joint velocities
            self.eff = None # joint accelerations
            
            self._status = "status"
            self._info = "info"
            self._warning = "warning"
            self._exception = "exception"

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "q")
            self.assign_views(cluster_aggregate, "v")
            self.assign_views(cluster_aggregate, "eff")

            self.last_index = self.offset
            
        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "q":
                
                # can only make views of contigous memory
                self.q = cluster_aggregate[:, self.offset:(self.offset + self._n_dofs)].view(self._cluster_size, 
                                                self._n_dofs)
                
                self.offset = self.offset + self._n_dofs
                
            if varname == "v":
                
                self.v = cluster_aggregate[:, self.offset:(self.offset + self._n_dofs)].view(self._cluster_size, 
                                                self._n_dofs)
                
                self.offset = self.offset + self._n_dofs

            if varname == "eff":
                
                self.eff = cluster_aggregate[:, self.offset:(self.offset + self._n_dofs)].view(self._cluster_size, 
                                                self._n_dofs)
                
                self.offset = self.offset + self._n_dofs

    class RhcInfo:

        def __init__(self,
                    cluster_aggregate: torch.Tensor, 
                    add_data_size: int, 
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.add_data_size = add_data_size

            self._cluster_size = cluster_aggregate.shape[0]

            self.info = None

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "info")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "info":
                
                self.info = cluster_aggregate[:, 
                                self.offset:(self.offset + self.add_data_size)].view(self._cluster_size, 
                                self.add_data_size)
                
                self.offset = self.offset + self.add_data_size

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                namespace = "",
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"),  
                dtype: torch.dtype = torch.float32, 
                add_data_size: int = 0):

        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.backend = "torch" # forcing torch backen
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_dofs = n_dofs
        
        self._terminate = False

        cluster_aggregate_columnsize = -1

        cluster_aggregate_columnsize = aggregate_cmd_size(self.n_dofs, 
                                                        add_data_size)
        
        self.cluster_aggregate = torch.zeros(
                                        (self.cluster_size, 
                                           cluster_aggregate_columnsize 
                                        ), 
                                        dtype=self.dtype, 
                                        device=self.device)
        
        self.jnt_cmd = self.JntCmd(self.cluster_aggregate,
                                    n_dofs = self.n_dofs, 
                                    prev_index=0)
        
        if not add_data_size == 0:
                                                        
            self.rhc_info = self.RhcInfo(self.cluster_aggregate,
                                    add_data_size, 
                                    prev_index=self.jnt_cmd.last_index)
        
        # this creates a shared memory block of the right size for the cmds
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=cmds_name(), 
                                    namespace=self.namespace,
                                    dtype=self.dtype) 

    def start(self):

        self.shared_memman.start() # will actually initialize the server

    def synch(self):

        # synchs jnt_cmd and rhc_info (which will normally live on GPU)
        # with the shared cmd data using the aggregate view (normally on CPU)

        # this requires a COPY FROM CPU TO GPU
        # (better to use the aggregate to exploit parallelization)

        self.cluster_aggregate[:, :] = self.shared_memman.tensor_view.cuda()

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

    def terminate(self):

        if not self._terminate:
            
            self._terminate = True

            self.shared_memman.terminate()

    def __del__(self):

        self.terminate()

class RhcClusterTaskRefs:

    class Phase:

        def __init__(self, 
                cluster_aggregate: torch.Tensor, 
                n_contacts: int, 
                prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.cluster_size = cluster_aggregate.shape[0]

            self.n_contacts = n_contacts

            self.phase_id = None # type of current phase (-1 custom, ...)
            self.is_contact = None # array of contact flags for each contact
            self.duration = None # phase duration
            self.p0 = None # start position
            self.p1 = None # end position
            self.clearance = None # flight clearance
            self.d0 = None # initial derivative
            self.d1 = None # end derivative

            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "phase_id")
            self.assign_views(cluster_aggregate, "is_contact")
            self.assign_views(cluster_aggregate, "duration")
            self.assign_views(cluster_aggregate, "p0")
            self.assign_views(cluster_aggregate, "p1")
            self.assign_views(cluster_aggregate, "clearance")
            self.assign_views(cluster_aggregate, "d0")
            self.assign_views(cluster_aggregate, "d1")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
            
            if varname == "phase_id":

                # (can only make views of contigous memory)
                
                self.phase_id = cluster_aggregate[:, self.offset].view(self.cluster_size, 
                                                                1)
                
                self.offset = self.offset + 1

            if varname == "is_contact":
                
                self.is_contact = cluster_aggregate[:, self.offset:(self.offset + self.n_contacts)].view(self.cluster_size, 
                                                            self.n_contacts)

                self.offset =  self.offset + self.n_contacts
            
            if varname == "duration":
                
                self.duration = cluster_aggregate[:, self.offset:(self.offset + 1)].view(self.cluster_size, 
                                                            1)

                self.offset =  self.offset + 1

            if varname == "p0":
                
                self.p0 = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                            3)

                self.offset =  self.offset + 3
            
            if varname == "p1":
                
                self.p1 = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                            3)

                self.offset =  self.offset + 3
            
            if varname == "clearance":
                
                self.clearance = cluster_aggregate[:, self.offset:(self.offset + 1)].view(self.cluster_size, 
                                                            1)

                self.offset =  self.offset + 1
            
            if varname == "d0":
                
                self.d0 = cluster_aggregate[:, self.offset:(self.offset + 1)].view(self.cluster_size, 
                                                            1)

                self.offset =  self.offset + 1

            if varname == "d1":
                
                self.d1 = cluster_aggregate[:, self.offset:(self.offset + 1)].view(self.cluster_size, 
                                                            1)

                self.offset =  self.offset + 1

    class BasePose:

        def __init__(self, 
                    cluster_aggregate: torch.Tensor,
                    prev_index: int = 0):
            
            self.prev_index = prev_index
            self.last_index = -1

            self.cluster_size = cluster_aggregate.shape[0]
        
            self.p = None # base position
            self.q = None # base orientation (quaternion)
            self.pose = None # full pose [p, q]

            self.offset = self.prev_index
            
            self.assign_views(cluster_aggregate, "pose")
            self.assign_views(cluster_aggregate, "p")
            self.assign_views(cluster_aggregate, "q")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
                        
            if varname == "pose":
                
                self.pose = cluster_aggregate[:, self.offset:(self.offset + 7)].view(self.cluster_size, 
                                                                    7)
                
            if varname == "p":
                
                self.p = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                                    3)
                
                self.offset = self.offset + 3
                
            if varname == "q":
                
                self.q = cluster_aggregate[:, self.offset:(self.offset + 4)].view(self.cluster_size, 
                                                                    4)
                    
                self.offset = self.offset + 4

            self.last_index = self.offset

    class ComPos:

        def __init__(self, 
                    cluster_aggregate: torch.Tensor,
                    q_remapping: List[int] = None, 
                    prev_index: int = 0):

            self.prev_index = prev_index
            self.last_index = -1

            self.q_remapping = None

            if q_remapping is not None:
                self.q_remapping = torch.tensor(q_remapping)

            self.cluster_size = cluster_aggregate.shape[0]
        
            self.com_pos = None # full com position
            self.com_q = None # com orientation
            self.com_pose = None # com pose
            
            self.offset = self.prev_index

            self.assign_views(cluster_aggregate, "com_pose")
            self.assign_views(cluster_aggregate, "com_pos")
            self.assign_views(cluster_aggregate, "com_q")

            self.last_index = self.offset

        def assign_views(self, 
            cluster_aggregate: torch.Tensor,
            varname: str):
                            
            if varname == "com_pose":

                # (can only make views of contigous memory)
                
                self.com_pose = cluster_aggregate[:, self.offset:(self.offset + 7)].view(self.cluster_size, 
                                                7)
            
            if varname == "com_pos":

                # (can only make views of contigous memory)
                
                self.com_pos = cluster_aggregate[:, self.offset:(self.offset + 3)].view(self.cluster_size, 
                                                3)

                self.offset = self.offset + 3
                
            if varname == "com_q":

                # (can only make views of contigous memory)
                
                self.com_pos = cluster_aggregate[:, self.offset:(self.offset + 4)].view(self.cluster_size, 
                                                4)
                
                self.offset = self.offset + 4
            
    def __init__(self, 
                n_contacts: int, 
                cluster_size: int = 1, 
                namespace = "",
                backend: str = "torch", 
                device: torch.device = torch.device("cpu"), 
                dtype: torch.dtype = torch.float32):
        
        self.namespace = namespace

        self.journal = Journal()

        self.dtype = dtype

        self.backend = "torch" # forcing torch backend
        self.device = device
        if (self.backend != "torch"):

            self.device = torch.device("cpu")

        self.cluster_size = cluster_size
        self.n_contacts = n_contacts
        cluster_aggregate_columnsize = aggregate_refs_size(self.n_contacts)

        self._terminate = False

        self.cluster_aggregate = torch.zeros(
                                    (self.cluster_size, 
                                        cluster_aggregate_columnsize 
                                    ), 
                                    dtype=self.dtype, 
                                    device=self.device)

        # views of cluster_aggregate
        self.phase_id = self.Phase(cluster_aggregate=self.cluster_aggregate, 
                                    n_contacts=self.n_contacts, 
                                    prev_index = 0)
        
        self.base_pose = self.BasePose(cluster_aggregate=self.cluster_aggregate, 
                                    prev_index = self.phase_id.last_index)
        
        self.com_pose = self.ComPos(cluster_aggregate=self.cluster_aggregate, 
                                    prev_index = self.base_pose.last_index)
        
        # this creates a shared memory block of the right size for the state
        # and a corresponding view of it
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=task_refs_name(), 
                                    namespace=self.namespace,
                                    dtype=self.dtype)
    
    def start(self):

        self.shared_memman.start() # will actually initialize the server

    def synch(self):

        # synchs jnt_cmd and rhc_info (which will normally live on GPU)
        # with the shared cmd data using the aggregate view (normally on CPU)

        # this requires a COPY FROM CPU TO GPU
        # (better to use the aggregate to exploit parallelization)

        self.cluster_aggregate[:, :] = self.shared_memman.tensor_view.cuda()

        torch.cuda.synchronize() # this way we ensure that after this the state on GPU
        # is fully updated

    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self.shared_memman.terminate()

    def __del__(self):
        
        self.terminate()

class HanshakeDataCntrlSrvr:

    def __init__(self, 
                verbose = False, 
                namespace = ""):
        
        # for now we use the wait amount to make race conditions practically 
        # impossible 
        
        self.verbose = verbose

        self.namespace = namespace

        self.journal = Journal()

        self.handshake_done = False
        self._terminate = False

        self.wait_amount = 0.1

        self.cluster_size = None
        self.jnt_names_client = None
        self.jnt_number_client = None
        self.add_data_length = None
        self.n_contacts = None

        self.cluster_size = SharedMemClient(name=cluster_size_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        
        self.jnt_number_client = SharedMemClient(name=jnt_number_client_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        
    def handshake(self):
        
        # first of all, we need to know the size of the cluster
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": executing handshake")

        self.cluster_size.attach()
        self.jnt_number_client.attach()

        self.jnt_names_client = SharedStringArray(length=self.jnt_number_client.tensor_view[0, 0].item(), 
                                    name=jnt_names_client_name(), 
                                    namespace=self.namespace,
                                    is_server=False, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        self.jnt_names_client.start()

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": handshake terminated")

        self.handshake_done = True

    def finalize_init(self, 
                add_data_length: int, 
                n_contacts: int):
        
        if self.handshake_done:
            # these are steps to be performed after the controllers are fully initialized

            # we create the clients (will wait for the memory to be 
            # created by the server)
            print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + \
                f"[{self.finalize_init.__name__}]" + ": executing finalization steps...")
            
            # we first create the servers (non-blocking)

            self.add_data_length = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=additional_data_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64)
            self.add_data_length.start()
            self.add_data_length.tensor_view[0, 0] = add_data_length

            self.n_contacts = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=n_contacts_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64)
            self.n_contacts.start()
            self.n_contacts.tensor_view[0, 0] = n_contacts

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + \
                f"[{self.finalize_init.__name__}]" + ": done.")
            
        else:

            exception = f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + \
                    f"{self.finalize_init.__name__}" + ": did you remember to call handshake() before?"
        
            raise Exception(exception)
        
    def terminate(self):
       
        if not self._terminate:

            self._terminate = True

            if self.cluster_size is not None:

                self.cluster_size.terminate()
            
            if self.jnt_names_client is not None:

                self.jnt_names_client.terminate()

            if self.jnt_number_client is not None:

                self.jnt_number_client.terminate()

            if self.add_data_length is not None:

                self.add_data_length.terminate()
            
            if self.n_contacts is not None:

                self.n_contacts.terminate()

    def __del__(self):

        self.terminate()

class HanshakeDataCntrlClient:

    def __init__(self, 
            n_jnts: int, 
            namespace = ""):
        
        # for now we use the wait amount to make race conditions practically 
        # impossible 

        self.n_jnts = n_jnts

        self.namespace = namespace

        self.journal = Journal()

        self.wait_amount = 0.1

        self.jnt_names_client = SharedStringArray(length=self.n_jnts, 
                                    name=jnt_names_client_name(), 
                                    namespace=self.namespace,
                                    is_server=True)

        self.cluster_size = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64)

        self.jnt_number_client = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=jnt_number_client_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64)

        self.add_data_length = SharedMemClient(name=additional_data_name(),  
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        
        self.n_contacts = SharedMemClient(name=n_contacts_name(),  
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        
        self._terminate = False

        self.handshake_done = False

    def start(self, 
            cluster_size: int, 
            jnt_names: List[str]):

        self._handshake(cluster_size,
                    jnt_names)
        
    def _handshake(self, 
                cluster_size: int, 
                jnt_names: List[str]):
        
        # first of all, we need to know the size of the cluster
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": executing handshake")

        # start servers
        if len(jnt_names) != self.n_jnts:

            exception = f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                + f"[{self._handshake.__name__}]" +  f": provided jnt names lenght {len(jnt_names)} does not match {self.n_jnts}"

            raise Exception(exception)
        
        self.jnt_names_client.start(init=jnt_names) # start server
        self.jnt_names_client.write(jnt_names)

        self.cluster_size.start() # start server and immediately write value to it
        self.cluster_size.tensor_view[0, 0] = cluster_size

        self.jnt_number_client.start()
        self.jnt_number_client.tensor_view[0, 0] = self.n_jnts

        # start clients

        self.add_data_length.attach()
        self.n_contacts.attach()

        self.handshake_done = True

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": handshake terminated")

    def terminate(self):
                
        if not self._terminate:
            
            self._terminate = True

            self.jnt_names_client.terminate() 

            self.cluster_size.terminate()

            self.jnt_number_client.terminate()

            self.add_data_length.terminate() # exists the initialiation loop, if still running

            self.n_contacts.terminate()

    def __del__(self):

        self.terminate()

# new SharsorIPC-based shared data
        
class RHCStatus():

    class ActivationStateView(SharedDataView):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "ClusterActivationState" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                dtype=dtype.Bool)
            
    class FailFlagView(SharedDataView):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "ClusterFailFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                dtype=dtype.Bool)
    
    class ResetFlagView(SharedDataView):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "ClusterResetFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                dtype=dtype.Bool)
    
    class TriggerFlagView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "ClusterTriggerFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                dtype=dtype.Bool)
            
    def __init__(self, 
            is_server = False, 
            cluster_size: int = -1, 
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0):

        self.is_server = is_server

        self.cluster_size = cluster_size

        self.namespace = namespace

        self.verbose = verbose

        self.vlevel = vlevel

        self.active = self.ActivationStateView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel)

        self.fails = self.FailFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel)
        
        self.resets = self.ResetFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel)
        
        self.trigger = self.TriggerFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel)
        
    def __del__(self):

        self.close()

    def run(self):
    
        # self.fails.run()
        # self.resets.run()
        self.trigger.run()
        # self.active.run()

    def close(self):
        
        self.trigger.close()

        # self.active.close()
        # self.fails.close()
        # self.resets.close()
                    
class JntImpCntrlData:

    class PosErrView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "PosErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class VelErrView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "VelErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class PosGainsView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "PosGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)
    
    class VelGainsView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "VelGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class PosView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "Pos" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class VelView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "Vel" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class EffView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "Eff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class PosRefView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "PosRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class VelRefView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "VelRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class EffFFView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "EffFF" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    class ImpEffView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0):
            
            basename = "ImpEff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel)

    def __init__(self, 
            is_server = False, 
            n_envs: int = -1, 
            n_jnts: int = -1,
            jnt_names: List[str] = [""],
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0):

        self.is_server = is_server

        self.n_envs = n_envs
        self.n_jnts = n_jnts

        self.jnt_names = jnt_names

        self.verbose = verbose
        self.vlevel = vlevel

        if self.is_server:

            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = "SharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel, 
                                        force_reconnection = True)

        else:

            self.shared_jnt_names = StringTensorClient(
                                        basename = "SharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel)

        self.pos_err_view = self.PosErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)
        
        self.vel_err_view = self.VelErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)
        
        self.pos_gains_view = self.PosGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.vel_gains_view = self.VelGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.eff_ff_view = self.EffFFView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.pos_view = self.PosView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)
        
        self.pos_ref_view = self.PosRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.vel_view = self.VelView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.vel_ref_view = self.VelRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.eff_view = self.EffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

        self.imp_eff_view = self.ImpEffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel)

    def __del__(self):

        self.terminate()

    def run(self):
        
        self.pos_err_view.run()
        self.vel_err_view.run()
        self.pos_gains_view.run()
        self.vel_gains_view.run()
        self.eff_ff_view.run()
        self.pos_view.run()
        self.pos_ref_view.run()
        self.vel_view.run()
        self.vel_ref_view.run()
        self.eff_view.run()
        self.imp_eff_view.run()

        # in case we are clients
        self.n_envs = self.pos_err_view.n_rows
        self.n_jnts = self.pos_err_view.n_cols

        # retrieving joint names
        self.shared_jnt_names.run()

        if self.is_server:

            jnt_names_written = self.shared_jnt_names.write_vec(self.jnt_names, 0)

            if not jnt_names_written:

                raise Exception("Could not write joint names on shared memory!")
        
        else:
            
            self.jnt_names = [""] * self.n_jnts

            jnt_names_read = self.shared_jnt_names.read_vec(self.jnt_names, 0)

            if not jnt_names_read:

                raise Exception("Could not read joint names on shared memory!")

    def terminate(self):
        
        self.pos_err_view.close()
        self.vel_err_view.close()
        self.pos_gains_view.close()
        self.vel_gains_view.close()
        self.eff_ff_view.close()
        self.pos_view.close()
        self.pos_ref_view.close()
        self.vel_view.close()
        self.vel_ref_view.close()
        self.eff_view.close()
        self.imp_eff_view.close()

        self.shared_jnt_names.close()

class RHC2SharedNamings:

    def __init__(self, 
            basename: str, 
            namespace: str, 
            index: int):

        self.index = index

        self.basename = basename
        self.namespace = namespace

        self.global_ns = f"{basename}_{namespace}_{self.index}"

        self.ROBOT_Q_NAME = "robot_q"
        self.RHC_Q_NAME = "rhc_q"

    def global_ns(self, 
            basename: str, 
            namespace: str):

        return f"{basename}_{namespace}"
    
    def get_robot_q_name(self):
        
        return f"{self.global_ns}_{self.ROBOT_Q_NAME}"
    
    def get_rhc_q_name(self):

        return f"{self.global_ns}_{self.RHC_Q_NAME}"

