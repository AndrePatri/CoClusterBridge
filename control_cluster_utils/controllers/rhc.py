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
        self._to_horizon_quat = [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
        self._jnt_maps_created = False

        self._init()

        self._pipe_opened = False

        self.array_dtype = array_dtype

    def _init(self):

        self._init_problem() # we call the child's initialization method

        self._init_states() # know that the n_dofs are known, we can call the parent method to init robot states and cmds

    def _open_pipes(self):
        
        # these are blocking
        self.pipes_manager.open_pipes(selector=["trigger_solve"
                ], 
                mode = OMode["O_RDONLY_NONBLOCK"], 
                index=self.controller_index)

        self.pipes_manager.open_pipes(selector=[ 
                "state_root_p", "state_root_q", "state_root_v", "state_root_omega", 
                "state_jnt_q", "state_jnt_v"
                ], 
                mode = OMode["O_RDONLY_NONBLOCK"], 
                index=self.controller_index)
        
        # these are blocking (we read even if the pipe is empty)
        self.pipes_manager.open_pipes(selector=["success", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "rhc_info"
                ], 
                mode = OMode["O_WRONLY"], 
                index=self.controller_index)
        
    def _close_pipes(self):

        # we close the pipes
        self.pipes_manager.close_pipes(selector=["trigger_solve", 
                "state_root_p", "state_root_q", "state_root_v", "state_root_omega", 
                "state_jnt_q", "state_jnt_v", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "state_jnt_q", "state_jnt_v", 
                "rhc_info"
                ], 
                index=self.controller_index)
    
    def _init_states(self):
        
        # to be called after n_dofs is known
        self.robot_state = RobotState(n_dofs=self.n_dofs, 
                                    cluster_size=self.cluster_size,
                                    index=self.controller_index,
                                    dtype=self.array_dtype, 
                                    verbose = self._verbose) # used for storing state coming FROM robot

        self.robot_cmds = RobotCmds(self.n_dofs, 
                                cluster_size=self.cluster_size, 
                                index=self.controller_index,
                                add_info_size=2, 
                                dtype=self.array_dtype, 
                                verbose=self._verbose) # used for storing internal state (i.e. from TO solution)
        
        self._init_sizes() # to make reading from pipes easier

    def update(self, 
               current_robot_state: RobotStateChild = None):

        # updates the internal state of the RHC controller. 
        # this can be done integrating the current internal state
        # with the computed actions or through sensory feedback
        
        success = False

        if current_robot_state is not None:
            
            success = self._update_closed_loop(current_robot_state)

        else:

            success = self._update_open_loop()

        return success
    
    def get(self):
        
        # gets the current control command computed 
        # after the last call to solve

        return self.cntrl_cmd
    
    def _init_sizes(self):

        self.root_p_size = self.robot_state.root_state.p.shape[0] * \
                        self.robot_state.root_state.p.shape[1] * \
                        self.robot_state.root_state.p.element_size()
        
        self.root_q_size = self.robot_state.root_state.q.shape[0] * \
                        self.robot_state.root_state.q.shape[1] * \
                        self.robot_state.root_state.q.element_size()
        
        self.root_v_size = self.robot_state.root_state.v.shape[0] * \
                        self.robot_state.root_state.v.shape[1] * \
                        self.robot_state.root_state.v.element_size()
        
        self.root_omega_size = self.robot_state.root_state.omega.shape[0] * \
                        self.robot_state.root_state.omega.shape[1] * \
                        self.robot_state.root_state.omega.element_size()
        
        self.jnt_q_size = self.robot_state.jnt_state.q.shape[0] * \
                        self.robot_state.jnt_state.q.shape[1] * \
                        self.robot_state.jnt_state.q.element_size()
        
        self.jnt_v_size = self.robot_state.jnt_state.v.shape[0] * \
                        self.robot_state.jnt_state.v.shape[1] * \
                        self.robot_state.jnt_state.v.element_size()
        
    def _read_state(self):
        
        self.robot_state.root_state.p = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_p"][self.controller_index], 
                                                            self.root_p_size), 
                                        dtype=self.array_dtype).reshape(1, 
                                                            self.robot_state.root_state.p.shape[1])
        
        self.robot_state.root_state.q = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_q"][self.controller_index], 
                                                            self.root_q_size), 
                                        dtype=self.array_dtype)[self._to_horizon_quat].reshape(1, 
                                                            self.robot_state.root_state.q.shape[1])
        
        self.robot_state.root_state.v = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_v"][self.controller_index], 
                                                            self.root_v_size), 
                                        dtype=self.array_dtype).reshape(1, 
                                                            self.robot_state.root_state.v.shape[1])
        
        self.robot_state.root_state.omega = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_omega"][self.controller_index],
                                                            self.root_omega_size), 
                                        dtype=self.array_dtype).reshape(1, 
                                                            self.robot_state.root_state.omega.shape[1])
        
        self.robot_state.jnt_state.q = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_jnt_q"][self.controller_index], 
                                                            self.jnt_q_size), 
                                        dtype=self.array_dtype)[self._to_server].reshape(1, 
                                                                            self.robot_state.jnt_state.q.shape[1]) # with joint remapping to controller's order
        
        self.robot_state.jnt_state.v = torch.frombuffer(os.read(self.pipes_manager.pipes_fd["state_jnt_v"][self.controller_index], 
                                                            self.jnt_v_size), 
                                        dtype=self.array_dtype)[self._to_server].reshape(1,
                                                                            self.robot_state.jnt_state.v.shape[1]) # with joint remapping to controller's order

    def _send_solution(self):
        
        # writes commands from robot state
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_q"][self.controller_index], 
                self.robot_cmds.jnt_cmd.q[0, self._to_client].numpy().tobytes()) # with joint remapping to client's order
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_v"][self.controller_index], 
                self.robot_cmds.jnt_cmd.v[0, self._to_client].numpy().tobytes()) # with joint remapping to client's order
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_eff"][self.controller_index], 
                self.robot_cmds.jnt_cmd.effort[0, self._to_client].numpy().tobytes()) # with joint remapping to client's order

        # write additional info
        os.write(self.pipes_manager.pipes_fd["rhc_info"][self.controller_index], self.robot_cmds.slvr_state.info.numpy().tobytes())
    
    def _fill_cmds_from_sol(self):

        # get data from the solution and updated the view on the shared data
        self.robot_cmds.jnt_cmd.q[:, :] = self._get_cmd_jnt_q_from_sol()

        self.robot_cmds.jnt_cmd.v[:, :] = self._get_cmd_jnt_v_from_sol()

        self.robot_cmds.jnt_cmd.eff[:, :] = self._get_cmd_jnt_eff_from_sol()

        self.robot_cmds.slvr_state.info[:, :] = self._get_additional_slvr_info()

    def solve(self):
        
        if not self._jnt_maps_created:

            self._create_jnt_maps()

        if not self._pipe_opened:
            
            # we open here the pipes so that they are opened into 
            # the child process where the solve() is spawned

            self._open_pipes()

            self._pipe_opened = True

        if self._termination_flag.value:
            
            self.terminate()

        else:

            while True:
                
                # we are always listening for a trigger signal from the client 

                try:
                    
                    if self._verbose:

                            start = time.perf_counter()

                    msg_bytes = b'1'
                    t = time.perf_counter()
                    signal = os.read(self.pipes_manager.pipes_fd["trigger_solve"][self.controller_index], 
                                    len(msg_bytes)).decode().strip()
                    print(str(time.perf_counter()-t ))
                    if signal == '1':
                        
                        # read latest states from pipe 

                        self._read_state()

                        self._solve()

                        self._fill_cmds_from_sol() # we get data from the solution        
                        
                        self._send_solution() # writes solution on pipe

                        # print("cmd debug n." + str(self.controller_index) + "\n" + 
                        #         "q_cmd: " + str(self.robot_cmds.jnt_cmd.q) + "\n" + 
                        #         "v_cmd: " + str(self.robot_cmds.jnt_cmd.v) + "\n" + 
                        #         "eff_cmd: " + str(self.robot_cmds.jnt_cmd.effort))

                        # os.write(self.pipes_manager.pipes_fd["success"][self.controller_index], b"success\n")
                        
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
        
    def terminate(self):

        # self._close_pipes()
        
        return True
    
    def assign_client_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._client_side_jnt_names = jnt_names

        self._got_jnt_names_client = True

    def _assign_server_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._server_side_jnt_names = jnt_names

        self._got_jnt_names_server = True

    def _create_jnt_maps(self):

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


