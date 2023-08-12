import torch

import time 

from typing import List

from control_cluster_utils.utilities.shared_mem import SharedMemSrvr, SharedMemClient, SharedStringArray

from control_cluster_utils.utilities.defs import aggregate_cmd_size, aggregate_state_size
from control_cluster_utils.utilities.defs import states_name, cmds_name
from control_cluster_utils.utilities.defs import cluster_size_name, additional_data_name
from control_cluster_utils.utilities.defs import jnt_number_client_name, jnt_number_srvr_name
from control_cluster_utils.utilities.defs import jnt_names_client_name
from control_cluster_utils.utilities.defs import client_writing_name, srvr_writing_name

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

        self._terminate = False

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
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=states_name(), 
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
        
        self._terminate = False

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
        self.shared_memman = SharedMemSrvr(n_rows=self.cluster_size, 
                                    n_cols=cluster_aggregate_columnsize, 
                                    name=cmds_name(), 
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
                verbose = False):
        
        # for now we use the wait amount to make race conditions practically 
        # impossible 
        
        self.verbose = verbose
        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        self.handshake_done = False
        self._terminate = False

        self.wait_amount = 0.1

        self.cluster_size = None
        self.jnt_names_client = None
        self.jnt_number_client = None
        self.add_data_length = None

        self.cluster_size = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        
        self.jnt_number_client = SharedMemClient(n_rows=1, n_cols=1,
                                        name=jnt_number_client_name(), 
                                        dtype=torch.int64, 
                                        wait_amount=self.wait_amount, 
                                        verbose=self.verbose)
        
    def handshake(self):
        
        # first of all, we need to know the size of the cluster
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": executing handshake")

        self.cluster_size.attach()
        self.jnt_number_client.attach()

        self.jnt_names_client = SharedStringArray(length=self.jnt_number_client.tensor_view[0, 0].item(), 
                                    name=jnt_names_client_name(), 
                                    is_server=False, 
                                    wait_amount=self.wait_amount, 
                                    verbose=self.verbose)
        self.jnt_names_client.start()

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": handshake terminated")

        self.handshake_done = True

    def finalize_init(self, 
                add_data_length: int):
        
        if self.handshake_done:
            # these are steps to be performed after the controllers are fully initialized

            # we create the clients (will wait for the memory to be 
            # created by the server)
            print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + \
                f"{self.finalize_init.__name__}" + ": executing finalization steps")
            
            # we first create the servers (non-blocking)

            self.add_data_length = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=additional_data_name(), 
                                    dtype=torch.int64)
            self.add_data_length.start()
            self.add_data_length.tensor_view[0, 0] = add_data_length
        
        else:

            exception = f"[{self.__class__.__name__}]" + f"[{self.status}]" + \
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

    def __del__(self):

        self.terminate()

class HanshakeDataCntrlClient:

    def __init__(self, 
            n_jnts: int):
        
        # for now we use the wait amount to make race conditions practically 
        # impossible 

        self.n_jnts = n_jnts

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        self.handshake_done = False

        self.wait_amount = 0.1

        self.jnt_names_client = SharedStringArray(length=self.n_jnts, 
                                    name=jnt_names_client_name(), 
                                    is_server=True)

        self.cluster_size = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=cluster_size_name(), 
                                    dtype=torch.int64)

        self.jnt_number_client = SharedMemSrvr(n_rows=1, n_cols=1, 
                                    name=jnt_number_client_name(), 
                                    dtype=torch.int64)
        

        self.add_data_length = SharedMemClient(n_rows=1, n_cols=1, 
                                    name=additional_data_name(), 
                                    dtype=torch.int64, 
                                    wait_amount=self.wait_amount, 
                                    verbose=True)
        
        self._terminate = False

    def start(self, 
            cluster_size: int, 
            jnt_names: List[str]):

        self._handshake(cluster_size,
                    jnt_names)
        
    def _handshake(self, 
                cluster_size: int, 
                jnt_names: List[str]):
        
        # first of all, we need to know the size of the cluster
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": executing handshake")

        # start servers
        if len(jnt_names) != self.n_jnts:

            exception = f"[{self.__class__.__name__}]" + f"[{self.exception}]" + \
                + f"[{self._handshake.__name__}]" +  f": provided jnt names lenght {len(jnt_names)} does not match {self.n_jnts}"

            raise Exception(exception)
        self.jnt_names_client.start(init=jnt_names)

        self.cluster_size.start()
        self.cluster_size.tensor_view[0, 0] = cluster_size

        self.jnt_number_client.start()
        self.jnt_number_client.tensor_view[0, 0] = self.n_jnts

        # start clients

        self.add_data_length.attach()
        
        self.handshake_done = True

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": handshake terminated")

    def terminate(self):
                
        if not self._terminate:
            
            self._terminate = True

            self.jnt_names_client.terminate() 

            self.cluster_size.terminate()

            self.jnt_number_client.terminate()

            self.add_data_length.terminate() # exists the initialiation loop, if still running

    def __del__(self):

        self.terminate()
