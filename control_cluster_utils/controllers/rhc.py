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
            trigger_pipename: str,
            success_pipename: str, 
            jnt_q_pipename: str,
            jnt_v_pipename: str,
            jnt_eff_pipename: str,
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

        self.trigger_pipe = trigger_pipename
        self.success_pipe = success_pipename
        self.jnt_q_pipe = jnt_q_pipename
        self.jnt_v_pipe = jnt_v_pipename
        self.jnt_eff_pipe = jnt_eff_pipename
        
        self._open_pipes()
        
        self.rhc_cmd: RHCCmdChild = RHCCmd()

        self.cntrl_cmd: CntrlCmdChild =  CntrlCmd()

        self.n_dofs = None

        self.robot_state: RobotStateChild = None 
        
        self._init()

    def _init(self):

        self._init_problem()

    def _open_pipes(self):

        print(self.name + ": trying to open pipe @ " + self.trigger_pipe)
        self.trigger_pipe_fd = os.open(self.trigger_pipe,  os.O_RDONLY | os.O_NONBLOCK)

        print(self.name + ": trying to open pipe @ " + self.success_pipe)
        self.success_pipe_fd = os.open(self.success_pipe, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode
        
        print(self.name + ": trying to open pipe @ " + self.jnt_q_pipe)
        self.jnt_q_pipe_fd = os.open(self.jnt_q_pipe, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode

        print(self.name + ": trying to open pipe @ " + self.jnt_v_pipe)
        self.jnt_v_pipe_fd = os.open(self.jnt_v_pipe, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode

        print(self.name + ": trying to open pipe @ " + self.jnt_eff_pipe)
        self.jnt_eff_pipe_fd = os.open(self.jnt_eff_pipe, os.O_WRONLY) # this will block until
        # something opens the pipe in read mode

    def _close_pipes(self):

        # we close the pipes
        if os.path.exists(self.trigger_pipe):
                  
            os.close(self.trigger_pipe_fd)
            print(self.name + ": closed pipe @" + self.trigger_pipe)

        if os.path.exists(self.success_pipe):
            
            os.close(self.success_pipe_fd)
            print(self.name + ": closed pipe @" + self.success_pipe)

        if os.path.exists(self.jnt_q_pipe):
            
            os.close(self.jnt_q_pipe_fd)
            print(self.name + ": closed pipe @" + self.jnt_q_pipe)

        if os.path.exists(self.jnt_v_pipe):
            
            os.close(self.jnt_v_pipe_fd)
            print(self.name + ": closed pipe @" + self.jnt_v_pipe)

        if os.path.exists(self.jnt_eff_pipe):
            
            os.close(self.jnt_eff_pipe_fd)
            print(self.name + ": closed pipe @" + self.jnt_eff_pipe)

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

    def solve(self):

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

                        print(f"Solution time from {self.name} process: " + str(duration))

            except BlockingIOError:

                continue

        self._solve_exited = True
        
    def terminate(self):

        self._close_pipes()
        
RHChild = TypeVar('RHChild', bound='RHController')
CntrlCmdChild = TypeVar('CntrlCmdChild', bound='CntrlCmd')


