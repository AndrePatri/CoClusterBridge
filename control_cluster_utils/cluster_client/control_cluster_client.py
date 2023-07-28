import torch

from abc import ABC, abstractmethod

from control_cluster_utils.utilities.control_cluster_utils import RobotClusterState, RobotClusterCmd, ActionChild
from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
from control_cluster_utils.utilities.sysutils import PathsGetter

OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

import os
import struct

import time

import numpy as np

import multiprocess as mp

import ctypes

class ControlClusterClient(ABC):

    def __init__(self, 
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            backend: str = "torch", 
            device: str = "cpu"):
        
        self.n_dofs = mp.Value('i', -1)
        self.jnt_data_size = mp.Value('i', -1)
        self.cluster_size = cluster_size
        
        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs

        self._backend = backend
        self._device = device

        self._robot_states: RobotClusterState = None

        self._is_cluster_ready = mp.Value('b', False)
        self._trigger_solve = mp.Array('b', self.cluster_size)
        self._trigger_read = mp.Array('b', self.cluster_size)

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        paths = PathsGetter()
        self.pipes_config_path = paths.PIPES_CONFIGPATH

        self.pipes_manager = NamedPipesHandler(self.pipes_config_path)
        self.pipes_manager.create_buildpipes()
        self.pipes_manager.create_runtime_pipes(self.cluster_size) # we create the remaining pipes

        self._post_init_finished = False

        self.solution_time = -1.0
        self.n_sim_step_per_cntrl = -1
        self.solution_counter = 0
        self._compute_n_control_actions()

        self._spawn_processes()

    def _spawn_processes(self):
        
        # we spawn the handshake() to another process, 
        # so that it's not blocking
        self._connection_process = mp.Process(target=self._handshake, 
                                name = "ControlClusterClient_handshake")
        self._connection_process.start()
        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _handshake process")

        self._trigger_processes = []
        self._solread_processes = []

        for i in range(0, self.cluster_size):

            self._trigger_processes.append(mp.Process(target=self._trigger_solution, 
                                                    name = "ControlClusterClient_trigger" + str(i), 
                                                    args=(i, )), 
                                )
            print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": spawned _trigger_solution processes n." + str(i))
            self._trigger_processes[i].start()

    def _compute_n_control_actions(self):

        if self.cluster_dt < self.control_dt:

            print(f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + ": cluster_dt has to be >= control_dt")

            self.n_sim_step_per_cntrl = 1
        
        else:
            
            self.n_sim_step_per_cntrl = round(self.cluster_dt / self.control_dt)
            self.cluster_dt = self.control_dt * self.n_sim_step_per_cntrl

        message = f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": the cluster controllers will run at a rate of " + \
                str(1.0 / self.cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self.control_dt) + "Hz.\n" + \
                "Number of sim steps per control steps: " + str(self.n_sim_step_per_cntrl)

        print(message)
    
    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index is, e.g., the current simulation loop number (0-based)

        return (control_index+1) % self.n_sim_step_per_cntrl == 0

    def _open_pipes(self):

        for i in range(self.cluster_size):

            # state to controllers 
            self.state_root_q_pipe_fd[i] = os.open(self.state_root_q_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_root_v_pipe_fd[i] = os.open(self.state_root_v_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_jnt_q_pipe_fd[i] = os.open(self.state_jnt_q_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_jnt_v_pipe_fd[i] = os.open(self.state_jnt_v_pipenames[i], 
                                                    os.O_WRONLY)
            
    def _create_state_buffers(self):
        
        data_aux = np.zeros((1, 1), dtype=np.float32)
        self.float32_size = data_aux.itemsize
        
        self._cmd_q_buffer = mp.Array(ctypes.c_float, self.cluster_size * self.n_dofs.value)
        self._cmd_v_buffer = mp.Array(ctypes.c_float, self.cluster_size * self.n_dofs.value)
        self._cmd_eff_buffer = mp.Array(ctypes.c_float, self.cluster_size * self.n_dofs.value) 

        self._add_info_size = 2
        self._add_info_datasize = 2 * self.float32_size

        self._rhc_info_buffer = mp.Array(ctypes.c_float, self.cluster_size * self._add_info_size) 

    def _handshake(self):
        
        # THIS RUNS IN A CHILD PROCESS --> we perform the "handshake" with
        # the server: we exchange crucial info which has to be shared between 
        # them
        
        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": waiting for handshake with the ControlCluster server...")

        # retrieves some important configuration information from the server
        self.pipes_manager.open_pipes(["cluster_size"], 
                                    mode=OMode["O_WRONLY"])
        cluster_size_data = struct.pack('i', self.cluster_size)
        os.write(self.pipes_manager.pipes_fd["cluster_size"], cluster_size_data) # the server is listening -> we send the info we need

        self.pipes_manager.open_pipes(selector=["jnt_number"], 
                                mode=OMode["O_RDONLY"])
        jnt_number_raw = os.read(self.pipes_manager.pipes_fd["jnt_number"], DSize["int"])
        self.n_dofs.value = struct.unpack('i', jnt_number_raw)[0]

        import numpy as np
        data = np.zeros((self.n_dofs.value, 1), dtype=np.float32)
        self.float32_size = data.itemsize
        self.jnt_data_size.value = data.shape[0] * data.shape[1] * self.float32_size

        self._is_cluster_ready.value = True # we signal the main process
        # the connection is established

        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": friendship with ControlCluster server established.")

    def _trigger_solution(self, 
                        index: int):

        # solver
        self.pipes_manager.open_pipes(["trigger"], 
                                    mode=OMode["O_WRONLY"]) # blocking (non-blocking
        # would throw error if nothing has opened the pipe in read mode)
        
        while True: # we keep the process alive

            if self._trigger_solve[index]: # this is set by the parent process

                # Send a signal to perform the solution
         
                os.write(self.pipes_manager.pipes_fd["trigger"][index], b'solve\n')

                self._trigger_solve[index] = False # we will wait for next signal
                # from the main process
            
            else:

                continue

    def _read_solution(self, 
                    index: int):

        # these are not blocking
        self.pipes_manager.open_pipes(selector=["success", 
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "rhc_info"
                ], 
                mode = OMode["O_RDONLY_NONBLOCK"], 
                index=index)

        while True: # we keep the process alive

            if self._trigger_read[index]: # this is set by the parent process

                while True: # continue polling pipe until a success is read

                    try:

                        response = os.read(self.pipes_manager.pipes_fd["success"][index], 1024).decode().strip()

                        if response == "success":
                                
                            self._cmd_q_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)] = \
                                np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_q"][index], self.jnt_data_size.value), 
                                                dtype=np.float32).reshape((1, self.n_dofs.value)).flatten()
                        
                            self._cmd_v_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)] = \
                                np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_v"][index], self.jnt_data_size.value),
                                                dtype=np.float32).reshape((1, self.n_dofs.value)).flatten()
                            
                            self._cmd_eff_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)] = \
                                np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_eff"][index], self.jnt_data_size.value),
                                                dtype=np.float32).reshape((1, self.n_dofs.value)).flatten()
                            
                            self._rhc_info_buffer[(index * self._add_info_size):(index * self._add_info_size + self._add_info_size)] = \
                                np.frombuffer(os.read(self.pipes_manager.pipes_fd["rhc_info"][index], self._add_info_datasize),
                                                dtype=np.float32)
        
                            # print("received q" + str(index) + str(self._cmd_q_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)]))
                            # print("received v" + str(index) + str(self._cmd_v_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)]))
                            # print("received eff" + str(index) + str(self._cmd_eff_buffer[(index * self.n_dofs.value):(index * self.n_dofs.value + self.n_dofs.value)]))
                            
                            break

                        else:

                            print(f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + ": received invald response " +  
                                response + " from pipe " + self.pipes_manager.pipes["success"][index])

                    except BlockingIOError or SystemError:

                        continue # try again to read

                self._trigger_read[index] = False # we will wait for next signal
                # from the main process
            
            else:

                continue

    def _check_state_size(self, 
                        cluster_state: RobotClusterState):

        if cluster_state.n_dofs != self.n_dofs.value:

            return False
        
        if cluster_state.cluster_size != self.cluster_size:

            return False
        
        return True
    
    @abstractmethod
    def _check_cmd_size(self, 
                    cluster_cmd: ActionChild):
        
        pass

    @abstractmethod
    def _synch_controllers_from_cluster(self):

        # pushes all necessary data from the cluster (which interfaces with the environment)
        # to each controller, so that their internal state is updated

        pass

    @abstractmethod
    def _synch_cluster_from_controllers(self):

        # synch the cluster with the data in each controller: 
        # this might include, for example, computed control commands

        pass

    def _post_initialization(self):

        # self._open_pipes()
        # print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": pipe opening completed")
            
        self._create_state_buffers() # used  to store the data received by the pipes

        for i in range(0, self.cluster_size):

            self._solread_processes.append(mp.Process(target=self._read_solution, 
                                                    name = "ControlClusterClient_solread" + str(i), 
                                                    args=(i, )), 
                                )
            print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _read_solution processes n." + str(i))
            self._solread_processes[i].start()

        self._robot_states = RobotClusterState(self.n_dofs.value, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device) # from robot to controllers
        
        self._controllers_cmds = RobotClusterCmd(self.n_dofs.value, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device)
        
        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": initialized cluster state")

        self._post_init_finished = True

    def update(self, 
            cluster_state: RobotClusterState = None):
        
        if (cluster_state is not None) and (self._check_state_size(cluster_state)):

            self._robot_states = cluster_state

            self._synch_controllers_from_cluster() 

            return True
        
        if (cluster_state is None):
            
            self._synch_cluster_from_controllers() 

            return True
            
        else:

            return False 
    
    def _send_trigger(self):

        for i in range(0, self.cluster_size):

            self._trigger_solve[i] = True # we signal the triggger process n.{i} to trigger the solution 
            # of the associated controller

    def _read_sols(self):

        for i in range(self.cluster_size):
            
            self._trigger_read[i] = True

    def _wait_for_solutions(self):
        
        while not all(not value for value in self._trigger_read):

            continue
    
    def _fill_cluster_cmds(self):

        self._controllers_cmds.jnt_cmd.q = torch.frombuffer(self._cmd_q_buffer.get_obj(), 
                    dtype=self._controllers_cmds.dtype).reshape(self._controllers_cmds.cluster_size, 
                                                                self._controllers_cmds.n_dofs)
        
        self._controllers_cmds.jnt_cmd.v = torch.frombuffer(self._cmd_v_buffer.get_obj(), 
                    dtype=self._controllers_cmds.dtype).reshape(self._controllers_cmds.cluster_size, 
                                                                self._controllers_cmds.n_dofs)

        self._controllers_cmds.jnt_cmd.eff = torch.frombuffer(self._cmd_v_buffer.get_obj(), 
                    dtype=self._controllers_cmds.dtype).reshape(self._controllers_cmds.cluster_size, 
                                                                self._controllers_cmds.n_dofs)
        
        self._controllers_cmds.rhc_info.info = torch.frombuffer(self._rhc_info_buffer.get_obj(),
                    dtype=self._controllers_cmds.dtype).reshape(self._controllers_cmds.cluster_size, 
                                                                self._add_info_size)

    def solve(self):

        # solve all the TO problems in the control cluster

        if (self._is_cluster_ready.value and self._post_init_finished):

            start_time = time.time()
            
            self._send_trigger() # send signal to all controllers

            self._read_sols() # reads from all controllers' solutions (this will automatically updated a shared
            # data buffer)

            self._wait_for_solutions() # will wait until all controllers are done solving their RH-TO

            self._fill_cluster_cmds() # copies data to cluster command object
            
            self.solution_counter += 1

            self.solution_time = time.time() - start_time # we profile the whole solution pipeline

        if (self._is_cluster_ready.value and (not self._post_init_finished)):
            
            self._post_initialization() # perform post-initialization steps

            print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": post initialization steps performed")

        if not self._is_cluster_ready.value:

            print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": waiting connection to ControlCluster server")

    @abstractmethod
    def get(self):

        pass
    
    @abstractmethod
    def set_commands(self,  
                    cluster_cmd: ActionChild):

        pass
    
    def close(self):

        self.__del__()
        
    def __del__(self):
        
        if self._connection_process is not None:
            
            if self._connection_process.is_alive():
                    
                self._connection_process.terminate()  # Forcefully terminate the process
                
                print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": terminating child process " + str(self._connection_process.name))
            
                self._connection_process.join()

        for process in self._trigger_processes:

            if process.is_alive():
                    
                process.terminate()  # Forcefully terminate the process
                
                print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": terminating child process " + str(process.name))
            
                process.join()

        for process in self._solread_processes:

            if process.is_alive():
                    
                process.terminate()  # Forcefully terminate the process
                
                print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": terminating child process " + str(process.name))
            
                process.join()