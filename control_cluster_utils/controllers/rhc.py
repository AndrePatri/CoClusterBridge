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

from typing import List

class RobotState:

    class RootState:

        def __init__(self):
            
            self.p = np.zeros((3, 1), dtype=np.float32) # floating base position
            self.q = np.zeros((4, 1), dtype=np.float32) # floating base orientation (quaternion)
            self.v = np.zeros((3, 1), dtype=np.float32) # floating base linear vel
            self.omega = np.zeros((3, 1), dtype=np.float32) # floating base angular vel

    class JntState:

        def __init__(self, 
                    n_dofs: int):

            self.q = np.zeros((1, n_dofs), dtype=np.float32) # joint positions
            self.v = np.zeros((1, n_dofs), dtype=np.float32) # joint velocities
            self.a = np.zeros((1, n_dofs), dtype=np.float32) # joint accelerations
            self.effort = np.zeros((1, n_dofs), dtype=np.float32) # joint efforts

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
            verbose = False, 
            array_dtype = np.float32):
        
        self.controller_index = controller_index

        self.pipes_manager = copy.deepcopy(pipes_manager) # we make a (deep) copy

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

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
        
        self._client_side_jnt_names = []
        self._server_side_jnt_names = []
        self._got_jnt_names_client = False
        self._got_jnt_names_server = False

        self._to_server = []
        self._to_client = []
        self._jnt_maps_created = False

        self._init()

        self._pipe_opened = False

        self.array_dtype = array_dtype

    def _init(self):

        self._init_problem() # we call the child's initialization method

    def _open_pipes(self):
        
        # these are not blocking
        self.pipes_manager.open_pipes(selector=["trigger", 
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
        self.pipes_manager.close_pipes(selector=["trigger", 
                "state_root_p", "state_root_q", "state_root_v", "state_root_omega", 
                "state_jnt_q", "state_jnt_v", 
                "success", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "state_jnt_q", "state_jnt_v", 
                "rhc_info"
                ], 
                index=self.controller_index)
    
    def _init_states(self):
        
        # to be called after n_dofs is known
        self.robot_state = RobotState(self.n_dofs) # used for storing state coming FROM robot

        self.robot_cmds = RobotState(self.n_dofs, 
                                add_info_size=2) # used for storing internal state (i.e. from TO solution)
        
        self._init_sizes() # to make reading from pipes easier

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
    
    def _init_sizes(self):

        self.root_p_size = self.robot_state.root_state.p.shape[0] * \
                        self.robot_state.root_state.p.shape[1] * \
                        self.robot_state.root_state.p.itemsize
        
        self.root_q_size = self.robot_state.root_state.q.shape[0] * \
                        self.robot_state.root_state.q.shape[1] * \
                        self.robot_state.root_state.q.itemsize
        
        self.root_v_size = self.robot_state.root_state.v.shape[0] * \
                        self.robot_state.root_state.v.shape[1] * \
                        self.robot_state.root_state.v.itemsize
        
        self.root_omega_size = self.robot_state.root_state.omega.shape[0] * \
                        self.robot_state.root_state.omega.shape[1] * \
                        self.robot_state.root_state.omega.itemsize
        
        self.jnt_q_size = self.robot_state.jnt_state.q.shape[0] * \
                        self.robot_state.jnt_state.q.shape[1] * \
                        self.robot_state.jnt_state.q.itemsize
        
        self.jnt_v_size = self.robot_state.jnt_state.v.shape[0] * \
                        self.robot_state.jnt_state.v.shape[1] * \
                        self.robot_state.jnt_state.v.itemsize
        
    def _read_state(self):
        
        self.robot_state.root_state.p = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_p"][self.controller_index], 
                                                            self.root_p_size), 
                                        dtype=np.float32)
        
        self.robot_state.root_state.q = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_q"][self.controller_index], 
                                                            self.root_q_size), 
                                        dtype=np.float32)
        
        self.robot_state.root_state.v = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_v"][self.controller_index], 
                                                            self.root_v_size), 
                                        dtype=np.float32)
        
        self.robot_state.root_state.omega = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_root_omega"][self.controller_index],
                                                            self.root_omega_size), 
                                        dtype=np.float32)
        
        self.robot_state.jnt_state.q = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_jnt_q"][self.controller_index], 
                                                            self.jnt_q_size), 
                                        dtype=np.float32)[self._to_server] # with joint remapping to controller's order
        
        self.robot_state.jnt_state.v = np.frombuffer(os.read(self.pipes_manager.pipes_fd["state_jnt_v"][self.controller_index], 
                                                            self.jnt_v_size), 
                                        dtype=np.float32)[self._to_server] # with joint remapping to controller's order

    def _send_solution(self):
        
        # writes commands from robot state
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_q"][self.controller_index], 
                self.robot_cmds.jnt_state.q[self._to_client].tobytes()) # with joint remapping to client's order
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_v"][self.controller_index], 
                self.robot_cmds.jnt_state.v[self._to_client].tobytes()) # with joint remapping to client's order
        os.write(self.pipes_manager.pipes_fd["cmd_jnt_eff"][self.controller_index], 
                self.robot_cmds.jnt_state.effort[self._to_client].tobytes()) # with joint remapping to client's order

        # write additional info
        os.write(self.pipes_manager.pipes_fd["rhc_info"][self.controller_index], self.robot_cmds.slvr_state.info.tobytes())
    
    def _fill_cmds_from_sol(self):

        # get data from the solution
        self.robot_cmds.jnt_state.q = self._get_cmd_jnt_q_from_sol()
        self.robot_cmds.jnt_state.v = self._get_cmd_jnt_v_from_sol()
        self.robot_cmds.jnt_state.effort = self._get_cmd_jnt_eff_from_sol()

        self.robot_cmds.slvr_state.info = self._get_additional_slvr_info()

    def solve(self):
        
        if not self._jnt_maps_created:

            self._create_jnt_maps()

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

                    # read latest states from pipe 
                    self._read_state()

                    self._solve()

                    duration = time.time() - start
                    
                    self._fill_cmds_from_sol() # we get data from the solution

                    self._send_solution() # writes solution on pipe

                    os.write(self.pipes_manager.pipes_fd["success"][self.controller_index], b"success\n")

                    if self._verbose:

                        print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + f"[{self.status}]" + ":" + f"Solution time -> " + str(duration))

            except BlockingIOError:

                continue

        self._solve_exited = True
        
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
    def _get_cmd_jnt_q_from_sol(self) -> np.ndarray:

        pass

    abstractmethod
    def _get_cmd_jnt_v_from_sol(self) -> np.ndarray:

        pass
    
    abstractmethod
    def _get_cmd_jnt_eff_from_sol(self) -> np.ndarray:

        pass
    
    abstractmethod
    def _get_additional_slvr_info(self) -> np.ndarray:

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


