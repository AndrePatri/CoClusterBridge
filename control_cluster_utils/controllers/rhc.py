import numpy as np

from abc import ABC, abstractmethod

from typing import TypeVar

import os 

import time 

import multiprocess as mp

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

    def __init__(self, 
                n_dofs: int):

        self.root_state = RobotState.RootState()

        self.jnt_state = RobotState.JntState(n_dofs)

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
            trigger_pipename: str, # solver pipes
            success_pipename: str, 
            cmd_jnt_q_pipename: str, # commands to robot
            cmd_jnt_v_pipename: str,
            cmd_jnt_eff_pipename: str,
            state_root_q_pipename: str, # state from robot
            state_root_v_pipename: str, # state from robot
            state_jnt_q_pipename: str, # state from robot
            state_jnt_v_pipename: str, # state from robot
            termination_flag: mp.Value,
            name = "RHController",
            verbose = False):
        
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

        # solver 
        self.trigger_pipe = trigger_pipename
        self.success_pipe = success_pipename
        # commands to robot
        self.cmd_jnt_q_pipename = cmd_jnt_q_pipename
        self.cmd_jnt_v_pipename = cmd_jnt_v_pipename
        self.cmd_jnt_eff_pipename = cmd_jnt_eff_pipename        
        # state from robot
        self.state_root_q_pipename = state_root_q_pipename
        self.state_root_v_pipename = state_root_v_pipename
        self.state_jnt_q_pipename = state_jnt_q_pipename
        self.state_jnt_v_pipename = state_jnt_v_pipename

        self.rhc_cmd: RHCCmdChild = RHCCmd()

        self.cntrl_cmd: CntrlCmdChild =  CntrlCmd()

        self.n_dofs = None

        self.robot_state: RobotStateChild = None 
        
        self._init()

        self._pipe_opened = False
        
    def _init(self):

        self._init_problem()

    def _open_pipes(self):

        # solver
        print("[" + self.name + "]" + "[status]: trying to open pipe @ " + self.trigger_pipe)
        self.trigger_pipe_fd = os.open(self.trigger_pipe,  os.O_RDONLY | os.O_NONBLOCK)

        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.success_pipe)
        self.success_pipe_fd = os.open(self.success_pipe, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode
        
        # commands to robot
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.cmd_jnt_q_pipename)
        self.jnt_q_pipe_fd = os.open(self.cmd_jnt_q_pipename, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.cmd_jnt_v_pipename)
        self.jnt_v_pipe_fd = os.open(self.cmd_jnt_v_pipename, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.cmd_jnt_eff_pipename)
        self.jnt_eff_pipe_fd = os.open(self.cmd_jnt_eff_pipename, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode

        # state from robot
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.state_root_q_pipename)
        self.state_root_q_pipe_fd = os.open(self.state_root_q_pipename, os.O_RDONLY | os.O_NONBLOCK) 
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.state_root_v_pipename)
        self.state_root_v_pipe_fd = os.open(self.state_root_v_pipename, os.O_RDONLY | os.O_NONBLOCK)
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.state_jnt_q_pipename)
        self.state_jnt_q_pipe_fd = os.open(self.state_jnt_q_pipename, os.O_RDONLY | os.O_NONBLOCK)
        print("[" + self.name + "]"  + "[status]: trying to open pipe @ " + self.state_jnt_v_pipename)
        self.state_jnt_v_pipe_fd = os.open(self.state_jnt_v_pipename, os.O_RDONLY | os.O_NONBLOCK)

    def _close_pipes(self):

        # we close the pipes

        # solver
        if os.path.exists(self.trigger_pipe):
                  
            os.close(self.trigger_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.trigger_pipe)
        if os.path.exists(self.success_pipe):
            
            os.close(self.success_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.success_pipe)

        # commands to robot
        if os.path.exists(self.cmd_jnt_q_pipename):
            
            os.close(self.jnt_q_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.cmd_jnt_q_pipename)
        if os.path.exists(self.cmd_jnt_v_pipename):
            
            os.close(self.jnt_v_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.cmd_jnt_v_pipename)
        if os.path.exists(self.cmd_jnt_eff_pipename):
            
            os.close(self.jnt_eff_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.cmd_jnt_eff_pipename)

        # state from robot
        if os.path.exists(self.state_root_q_pipename):
            
            os.close(self.state_root_q_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.state_root_q_pipename)
        if os.path.exists(self.state_root_v_pipename):
            
            os.close(self.state_root_v_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.state_root_v_pipename)
        if os.path.exists(self.state_jnt_q_pipename):
            
            os.close(self.state_jnt_q_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.state_jnt_q_pipename)
        if os.path.exists(self.state_jnt_v_pipename):
            
            os.close(self.state_jnt_v_pipe_fd)
            print("[" + self.name + "]"  + "[status]: closed pipe @" + self.state_jnt_v_pipename)

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
            
            # we open here the pipe so that they are opened into 
            # the child process where the solve() is spawned

            self._open_pipes()

            self._pipe_opened = True

        while not self._termination_flag.value:

            try:

                signal = os.read(self.trigger_pipe_fd, 1024).decode().strip()

                if signal == 'terminate':
                    
                    self._solve_exited = True

                    break
                    
                elif signal == 'solve':
                                    
                    start = time.time()

                    self._solve()

                    duration = time.time() - start
                    
                    self._send_solution() # writes solution on pipe

                    os.write(self.success_pipe_fd, b"success\n")

                    if self._verbose:

                        print("[" + self.name + "]"  + "[status]:" + f"Solution time from {self.name} controller: " + str(duration))

            except BlockingIOError:

                continue

        self._solve_exited = True
        
    def terminate(self):

        # self._close_pipes()

        return True
        
RHChild = TypeVar('RHChild', bound='RHController')
CntrlCmdChild = TypeVar('CntrlCmdChild', bound='CntrlCmd')


