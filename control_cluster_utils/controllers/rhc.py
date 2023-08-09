import numpy as np

from abc import ABC, abstractmethod

from typing import TypeVar

import os 

import time 

import multiprocess as mp

from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

from control_cluster_utils.utilities.rhc_defs import RobotStateChild, RobotCmds, RobotState
from control_cluster_utils.utilities.shared_mem import SharedMemClient, SharedMemSrvr

import copy

from typing import List

import numpy as np
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
            pipes_manager: NamedPipesHandler,
            controller_index: int,
            termination_flag: mp.Value,
            verbose = False, 
            array_dtype = torch.float32):
        
        self.controller_index = controller_index

        self.pipes_manager = copy.deepcopy(pipes_manager) # we make a (deep) copy

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        self._termination_flag = termination_flag

        self._verbose = verbose
        
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

        self._pipe_opened = False

        self.array_dtype = array_dtype

    def _init(self):

        self._init_problem() # we call the child's initialization method

    def _open_pipes(self):
        
        # these are blocking
        self.pipes_manager.open_pipes(selector=["trigger_solve"
                ], 
                mode = OMode["O_RDONLY_NONBLOCK"], 
                index=self.controller_index)

        # these are blocking (we read even if the pipe is empty)
        self.pipes_manager.open_pipes(selector=["solved"], 
                mode = OMode["O_WRONLY"], 
                index=self.controller_index)
        
    def _close_pipes(self):

        # we close the pipes
        self.pipes_manager.close_pipes(selector=["trigger_solve", "solved"], 
                index=self.controller_index)
    
    def init_states(self):
        
        # to be called after n_dofs is known
        self.robot_state = RobotState(n_dofs=self.n_dofs, 
                                    cluster_size=self.cluster_size,
                                    index=self.controller_index,
                                    dtype=self.array_dtype,
                                    jnt_remapping=self._to_client, 
                                    q_remapping=self._quat_remap, 
                                    verbose = self._verbose) 

        self.robot_cmds = RobotCmds(self.n_dofs, 
                                cluster_size=self.cluster_size, 
                                index=self.controller_index,
                                add_info_size=2, 
                                dtype=self.array_dtype, 
                                jnt_remapping=self._to_server,
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
        
        if not self._pipe_opened:
            
            # we open here the pipes so that they are opened into 
            # the child process where the solve() is spawned

            self._open_pipes()

            self._pipe_opened = True

        if self._termination_flag.value:
            
            a = 1

        else:

            while True:
                
                # we are always listening for a trigger signal from the client 

                try:
                    
                    if self._verbose:
                            
                        start = time.perf_counter() 

                    msg_bytes = b'1'
                    signal = os.read(self.pipes_manager.pipes_fd["trigger_solve"][self.controller_index], 
                                    len(msg_bytes)).decode().strip()
                    
                    if signal == '1':
                        
                        # read latest states from pipe 

                        self._solve() # solve actual TO

                        self._fill_cmds_from_sol() # we upd update the views of the cmd
                        # from the solution
                        
                        # we signal the client this controller has finished its job
                        os.write(self.pipes_manager.pipes_fd["solved"][self.controller_index], 
                            b'1')
                        
                        if self._verbose:
                            
                            duration = time.perf_counter() - start

                            print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.info}]" + ":" + f"solve loop execution time  -> " + str(duration))
                        
                    else:

                        if self._verbose:

                            print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.warning}]" + ":" + f"received invalid signal {signal} on trigger solve pipe")
                            
                except BlockingIOError:

                    continue
    
    def assign_client_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._client_side_jnt_names = jnt_names

        self._got_jnt_names_client = True

    def _assign_server_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._server_side_jnt_names = jnt_names

        self._got_jnt_names_server = True

    def create_jnt_maps(self):

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


