import numpy as np

from abc import ABC, abstractmethod

from typing import TypeVar

import os 

import time 

import multiprocess as mp

from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

import copy

class RobotState:

    class RootState:

        def __init__(self):

            self.q = np.zeros((4, 1), dtype=np.float32) # floating base orientation (quaternion)
            self.v = np.zeros((3, 1), dtype=np.float32) # floating base angular vel
            self.a = np.zeros((3, 1), dtype=np.float32) # floating base angular acc

    class JntState:

        def __init__(self, 
                    n_dofs: int):

            self.q = np.zeros((n_dofs, 1), dtype=np.float32) # joint positions
            self.v = np.zeros((n_dofs, 1), dtype=np.float32) # joint velocities
            self.a = np.zeros((n_dofs, 1), dtype=np.float32) # joint accelerations
            self.effort = np.zeros((n_dofs, 1), dtype=np.float32) # joint efforts

    class SolverState:

        def __init__(self, 
                add_info_size = 1):

            self.info = np.zeros((add_info_size, 1), dtype=np.float32)

    def __init__(self, 
                n_dofs: int, 
                add_info_size: int = None):

        self.root_state = RobotState.RootState()

        self.jnt_state = RobotState.JntState(n_dofs)

        if add_info_size is not None:

            self.slvr_state = RobotState.SolverState(n_dofs)

        self.n_dofs = n_dofs

RobotStateChild = TypeVar('RobotStateChild', bound='RobotState')

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
            pipes_manager: NamedPipesHandler,
            controller_index: int,
            termination_flag: mp.Value,
            name = "RHController",
            verbose = False):
        
        self.controller_index = controller_index

        self.pipes_manager = copy.deepcopy(pipes_manager) # we make a copy

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        self.name = name

        self._termination_flag = termination_flag

        self._verbose = verbose
        
        self._solve_exited = False

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
        
        self._init()

        self._pipe_opened = False
        
    def _init(self):

        self._init_problem()

    def _open_pipes(self):
        
        # these are not blocking
        self.pipes_manager.open_pipes(selector=["trigger", 
                "state_root_p", "state_root_q", "state_root_v", "state_root_omega", 
                "state_jnt_q", "state_jnt_v"
                ], 
                mode = OMode["O_RDONLY_NONBLOCK"], 
                index=self.controller_index)

        # these are blocking
        self.pipes_manager.open_pipes(selector=["success", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "rhc_info"
                ], 
                mode = OMode["O_WRONLY"], 
                index=self.controller_index)
        
    def _close_pipes(self):

        # we close the pipes
        self.pipes_manager.close_pipes(selector=["trigger", 
                "state_root_p", "state_root_q", "state_root_v", "state_root_omega", 
                "state_jnt_q", "state_jnt_v", 
                "success", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "state_jnt_q", "state_jnt_v", 
                "rhc_info"
                ], 
                index=self.controller_index)
    
    @abstractmethod
    def _get_ndofs(self):

        pass

    @abstractmethod
    def _init_problem(self):

        # initialized horizon's TO problem

        pass
    
    def set_commands(self, 
                    action: RHCCmdChild):

        # sets all run-time parameters of the RHC controller:
        # command references, phases, etc...

        self.rhc_cmd = action
    
    def _update_open_loop(self):

        # updates measured robot state 
        # using the internal robot state of the RHC controller

        pass

    def _update_closed_loop(self, 
               current_robot_state: RobotStateChild):

        # updates measured robot state 
        # using the provided measurements

        self.robot_state = current_robot_state

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
    
    @abstractmethod
    def _solve(self):

        pass
    
    @abstractmethod
    def _send_solution(self):

        pass

    @abstractmethod
    def _acquire_state(self):

        pass

    def solve(self):
        
        if not self._pipe_opened:
            
            # we open here the pipes so that they are opened into 
            # the child process where the solve() is spawned

            self._open_pipes()

            self._pipe_opened = True

        while not self._termination_flag.value:

            try:

                signal = os.read(self.pipes_manager.pipes_fd["trigger"][self.controller_index], 1024).decode().strip()

                if signal == 'terminate':
                    
                    self._solve_exited = True

                    break
                    
                elif signal == 'solve':
                                    
                    start = time.time()

                    self._solve()

                    duration = time.time() - start
                    
                    self._send_solution() # writes solution on pipe

                    os.write(self.pipes_manager.pipes_fd["success"][self.controller_index], b"success\n")

                    if self._verbose:

                        print("[" + self.name + "]"  + f"[{self.status}]" + ":" + f"Solution time from {self.name} controller: " + str(duration))

            except BlockingIOError:

                continue

        self._solve_exited = True
        
    def terminate(self):

        # self._close_pipes()

        return True
        
RHChild = TypeVar('RHChild', bound='RHController')
CntrlCmdChild = TypeVar('CntrlCmdChild', bound='CntrlCmd')


