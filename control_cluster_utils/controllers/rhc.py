from abc import ABC, abstractmethod

import time 

from control_cluster_utils.utilities.rhc_defs import RobotStateChild, RobotCmds, RobotState
from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.defs import trigger_flagname

from typing import List, TypeVar

import torch


class CntrlCmd(ABC):

    pass

class RHCCmd(ABC):

    pass

RHCCmdChild = TypeVar('RHCCmdChild', bound='RHCCmd')

class RHController(ABC):

    def __init__(self, 
            urdf_path: str, 
            srdf_path: str,
            config_path: str, 
            cluster_size: int,
            controller_index: int,
            verbose = False, 
            debug = False,
            array_dtype = torch.float32):
        
        self.controller_index = controller_index

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        self._verbose = verbose
        self._debug = debug

        self.cluster_size = cluster_size

        self.urdf_path = urdf_path
        self.srdf_path = srdf_path
        # read urdf and srdf files
        with open(self.urdf_path, 'r') as file:

            self.urdf = file.read()
            
        with open(self.srdf_path, 'r') as file:

            self.srdf = file.read()

        self.config_path = config_path

        self.rhc_cmd: RHCCmdChild = RHCCmd()

        self.cntrl_cmd: CntrlCmdChild =  CntrlCmd()

        self.n_dofs = None

        self.robot_state: RobotStateChild = None 
        
        self._client_side_jnt_names = []
        self._server_side_jnt_names = []
        self._got_jnt_names_client = False
        self._got_jnt_names_server = False

        self._to_server = []
        self._to_client = []
        self._quat_remap = [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
        self._jnt_maps_created = False
        self._states_initialized = False

        self._init()

        self.array_dtype = array_dtype

        self.add_data_lenght = 0

    def __del__(self):
        
        self._terminate()

    def _terminate(self):

        self.trigger_flag.terminate()

        self.robot_cmds.terminate()
        self.robot_state.terminate()

    def _init(self):

        self._init_problem() # we call the child's initialization method

        dtype = torch.bool
        self.trigger_flag = SharedMemClient(n_rows=self.cluster_size, n_cols=1, 
                                    name=trigger_flagname(), 
                                    client_index=self.controller_index, 
                                    dtype=dtype)
        self.trigger_flag.attach()
    
    def init_states(self):
        
        # to be called after n_dofs is known
        self.robot_state = RobotState(n_dofs=self.n_dofs, 
                                    cluster_size=self.cluster_size,
                                    index=self.controller_index,
                                    dtype=self.array_dtype,
                                    jnt_remapping=self._to_server, 
                                    q_remapping=self._quat_remap, 
                                    verbose = self._verbose) 

        self.robot_cmds = RobotCmds(self.n_dofs, 
                                cluster_size=self.cluster_size, 
                                index=self.controller_index,
                                add_info_size=2, 
                                dtype=self.array_dtype, 
                                jnt_remapping=self._to_client,
                                verbose=self._verbose) 

        self._states_initialized = True
         
    def _fill_cmds_from_sol(self):

        # gets data from the solution and updates the view on the shared data
        
        self.robot_cmds.jnt_cmd.set_q(self._get_cmd_jnt_q_from_sol())

        self.robot_cmds.jnt_cmd.set_v(self._get_cmd_jnt_v_from_sol())

        self.robot_cmds.jnt_cmd.set_eff(self._get_cmd_jnt_eff_from_sol())

        self.robot_cmds.slvr_state.set_info(self._get_additional_slvr_info())

    def solve(self):
        
        if not self._jnt_maps_created:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.exception}]" + f"[{self.solve.__name__}]" + \
                                ":" + f"jnt maps not initialized. Did you call the create_jnt_maps()?"

            raise Exception(exception)

        if not self._states_initialized:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.exception}]" + f"[{self.solve.__name__}]" + \
                                ":" + f"states not initialized. Did you call the init_states()?"

            raise Exception(exception)

        else:

            while True:
                
                # we are always listening for a trigger signal from the client 

                try:
                    
                    if self._debug:
                            
                        start = time.perf_counter() 
                    
                    if self.trigger_flag.read_bool():
                        
                        # read latest states from pipe 

                        self._solve() # solve actual TO

                        self._fill_cmds_from_sol() # we upd update the views of the cmd
                        # from the solution
                        
                        # we signal the client this controller has finished its job
                        self.trigger_flag.set_bool(False) # this is also necessary to trigger again the solution
                        # on next loop, unless the client requires it
                        
                        if self._debug:
                            
                            duration = time.perf_counter() - start

                        # if self._verbose and self._debug:

                        print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                            f"[{self.info}]" + ":" + f"solve loop execution time  -> " + str(duration))
                        
                    # else:

                    #     if self._verbose:
                            
                    #         print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                    #             f"[{self.warning}]" + ":" + f" waiting for solution trigger...")

                except KeyboardInterrupt:

                    break
    
    def assign_client_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._client_side_jnt_names = jnt_names
        
        self._got_jnt_names_client = True

    def _assign_server_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._server_side_jnt_names = jnt_names

        self._got_jnt_names_server = True

    def _check_jnt_names_compatibility(self):

        set_srvr = set(self._server_side_jnt_names)
        set_client  = set(self._client_side_jnt_names)

        if not set_srvr == set_client:

            exception = f"[{self.__class__.__name__}]" + f"{self.controller_index}" + f"{self.exception}" + \
                ": server-side and client-side joint names do not match!"

            raise Exception(exception)
        
    def create_jnt_maps(self):

        self._check_jnt_names_compatibility() # will raise exception

        if not self._got_jnt_names_client:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                f"[{self.exception}]" + ":" + f"Cannot run the solve().  assign_client_side_jnt_names() was not called!"

            raise Exception(exception)
        
        if not self._got_jnt_names_server:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                f"[{self.exception}]" + ":" + f"Cannot run the solve().  _assign_server_side_jnt_names() was not called!"

            raise Exception(exception)
        
        self._to_server = [self._client_side_jnt_names.index(element) for element in self._server_side_jnt_names]

        self._to_client = [self._server_side_jnt_names.index(element) for element in self._client_side_jnt_names]
        
        self._jnt_maps_created = True

    @abstractmethod
    def _get_robot_jnt_names(self) -> List[str]:

        pass

    @abstractmethod
    def _get_cmd_jnt_q_from_sol(self) -> torch.Tensor:

        pass

    @abstractmethod
    def _get_cmd_jnt_v_from_sol(self) -> torch.Tensor:

        pass
    
    @abstractmethod
    def _get_cmd_jnt_eff_from_sol(self) -> torch.Tensor:

        pass
    
    @abstractmethod
    def _get_additional_slvr_info(self) -> torch.Tensor:

        pass

    @abstractmethod
    def _update_open_loop(self):

        # updates measured robot state 
        # using the internal robot state of the RHC controller

        pass
    
    @abstractmethod
    def _update_closed_loop(self):

        # updates measured robot state 
        # using the provided measurements

        pass

    @abstractmethod
    def _solve(self):

        pass
            
    @abstractmethod
    def _get_ndofs(self):

        pass

    @abstractmethod
    def _init_problem(self):

        # initialized horizon's TO problem

        pass
   
RHChild = TypeVar('RHChild', bound='RHController')
CntrlCmdChild = TypeVar('CntrlCmdChild', bound='CntrlCmd')


