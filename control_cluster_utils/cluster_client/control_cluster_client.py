import torch

from abc import ABC, abstractmethod

from control_cluster_utils.utilities.control_cluster_utils import RobotClusterState, ActionChild

import os
import struct

import time

import numpy as np

import multiprocess as mp

class ControlClusterClient(ABC):

    def __init__(self, 
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            backend: str = "torch", 
            device: str = "cpu"):
        
        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs

        self.n_dofs = mp.Value('i', -1)
        self.jnt_data_size = mp.Value('i', -1)
        self.cluster_size = cluster_size

        self._backend = backend

        self._device = device

        self._robot_states: RobotClusterState = None

        self.pipe_basepath = "/tmp/control_cluster_pipes/"
        
        self.cluster_size_pipe = None
        self.cluster_size_pipe_fd_mp = mp.Value('i', -1)
        self.jnt_number_pipe = None
        self.jnt_number_pipe_fd_mp = mp.Value('i', -1)

        # we create several named pipes and store their names

        self.trigger_pipenames = []
        self.success_pipenames = []
        self.cmd_jnt_q_pipenames = []
        self.cmd_jnt_v_pipenames = []
        self.cmd_jnt_eff_pipenames = []
        self.state_root_q_pipenames = []
        self.state_root_v_pipenames = []
        self.state_jnt_q_pipenames = []
        self.state_jnt_v_pipenames = []

        self.success_pipes_fd = [-1] * self.cluster_size 
        self.cmd_jnt_q_pipes_fd = [-1] * self.cluster_size 
        self.cmd_jnt_v_pipes_fd = [-1] * self.cluster_size 
        self.cmd_jnt_eff_pipes_fd = [-1] * self.cluster_size 
        self.state_root_q_pipe_fd = [-1] * self.cluster_size 
        self.state_root_v_pipe_fd = [-1] * self.cluster_size 
        self.state_jnt_q_pipe_fd = [-1] * self.cluster_size 
        self.state_jnt_v_pipe_fd = [-1] * self.cluster_size 
        
        self._is_cluster_ready = mp.Value('b', False)
        self._trigger_solve = mp.Array('b', self.cluster_size)

        self._pipes_initialized = False
        self._pipes_created = False

        self.solution_time = -1.0
        self.n_sim_step_per_cntrl = -1
        self.solution_counter = 0
        self._compute_n_control_actions()

        self._create_pipes() 

        self._spawn_processes()

    def _spawn_processes(self):
        
        # we spawn the handshake() to another process, 
        # so that it's not blocking
        self._connection_process = mp.Process(target=self._handshake, 
                                name = "ControlClusterClient_handshake")
        self._connection_process.start()
        print("[ControlClusterClient][status]: spawned _handshake process")

        # we also spawn the method to trigger 

        self._trigger_processes = []
        for i in range(0, self.cluster_size):
            self._trigger_processes.append(mp.Process(target=self._trigger_solution, 
                                                    name = "ControlClusterClient_trigger" + str(i), 
                                                    args=(i, )), 
                                )
            print("[ControlClusterClient][status]: spawned _trigger_solution processes n." + str(i))
            self._trigger_processes[i].start()

    def _compute_n_control_actions(self):

        if self.cluster_dt < self.control_dt:

            print("[ControlClusterClient][warning]: cluster_dt has to be >= control_dt")

            self.n_sim_step_per_cntrl = 1
        
        else:
            
            self.n_sim_step_per_cntrl = round(self.cluster_dt / self.control_dt)
            self.cluster_dt = self.control_dt * self.n_sim_step_per_cntrl

        message = "[ControlClusterClient][info]: the cluster controllers will run at a rate of " + \
                str(1.0 / self.cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self.control_dt) + "Hz.\n" + \
                "Number of sim steps per control steps: " + str(self.n_sim_step_per_cntrl)

        print(message)
    
    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index is, e.g., the current simulation loop number (0-based)

        return (control_index+1) % self.n_sim_step_per_cntrl == 0

    def _create_pipes(self):

        if not os.path.exists(self.pipe_basepath):
            
            print("[ControlClusterClient][status]: creating pipes directory @ " + self.pipe_basepath)
            os.mkdir(self.pipe_basepath)

        self.cluster_size_pipe = self.pipe_basepath + f"cluster_size.pipe"
        if not os.path.exists(self.cluster_size_pipe):
            
            print("[ControlClusterClient][status]: creating pipe @ " + self.cluster_size_pipe)
            os.mkfifo(self.cluster_size_pipe)

        self.jnt_number_pipe = self.pipe_basepath + f"jnt_number.pipe"

        if not os.path.exists(self.jnt_number_pipe):
            
            print("[ControlClusterClient][status]: creating pipe @" + self.jnt_number_pipe)
            os.mkfifo(self.jnt_number_pipe)

        for i in range(self.cluster_size):
            
            # solver 
            trigger_pipename = self.pipe_basepath + f"trigger{i}.pipe"
            success_pipename = self.pipe_basepath + f"success{i}.pipe"
            self.trigger_pipenames.append(trigger_pipename)
            self.success_pipenames.append(success_pipename)

            if not os.path.exists(trigger_pipename):
                
                print("[ControlClusterClient][status]: creating pipe @" + trigger_pipename)
                os.mkfifo(trigger_pipename)

            if not os.path.exists(success_pipename):
                
                print("[ControlClusterClient][status]: creating pipe @" + success_pipename)
                os.mkfifo(success_pipename)
        
            # cmds from controllers
            cmd_jnt_q_pipename = self.pipe_basepath + f"cmd_jnt_q{i}.pipe"
            cmd_jnt_v_pipename = self.pipe_basepath + f"cmd_jnt_v{i}.pipe"
            cmd_jnt_eff_pipename = self.pipe_basepath + f"cmd_jnt_eff{i}.pipe"
            self.cmd_jnt_q_pipenames.append(cmd_jnt_q_pipename)
            self.cmd_jnt_v_pipenames.append(cmd_jnt_v_pipename)
            self.cmd_jnt_eff_pipenames.append(cmd_jnt_eff_pipename)

            if not os.path.exists(cmd_jnt_q_pipename):
        
                print("[ControlClusterClient][status]: creating pipe @" + cmd_jnt_q_pipename)
                os.mkfifo(cmd_jnt_q_pipename)

            if not os.path.exists(cmd_jnt_v_pipename):
                
                print("[ControlClusterClient][status]: creating pipe @" + cmd_jnt_v_pipename)
                os.mkfifo(cmd_jnt_v_pipename)

            if not os.path.exists(cmd_jnt_eff_pipename):
                
                print("[ControlClusterClient][status]: creating pipe @" + cmd_jnt_eff_pipename)
                os.mkfifo(cmd_jnt_eff_pipename)
            
            # state to controllers 
            state_root_q_pipename = self.pipe_basepath + f"state_root_q{i}.pipe"
            state_root_v_pipename = self.pipe_basepath + f"state_root_v{i}.pipe"
            state_jnt_q_pipename = self.pipe_basepath + f"state_jnt_q{i}.pipe"
            state_jnt_v_pipename = self.pipe_basepath + f"state_jnt_v{i}.pipe"
            self.state_root_q_pipenames.append(state_root_q_pipename)
            self.state_root_v_pipenames.append(state_root_v_pipename)
            self.state_jnt_q_pipenames.append(state_jnt_q_pipename)
            self.state_jnt_v_pipenames.append(state_jnt_v_pipename)
            
            if not os.path.exists(state_root_q_pipename):
        
                print("[ControlClusterClient][status]: creating pipe @" + state_root_q_pipename)
                os.mkfifo(state_root_q_pipename)
            
            if not os.path.exists(state_root_v_pipename):
        
                print("[ControlClusterClient][status]: creating pipe @" + state_root_v_pipename)
                os.mkfifo(state_root_v_pipename)
            
            if not os.path.exists(state_jnt_q_pipename):
        
                print("[ControlClusterClient][status]: creating pipe @" + state_jnt_q_pipename)
                os.mkfifo(state_jnt_q_pipename)
            
            if not os.path.exists(state_jnt_v_pipename):
        
                print("[ControlClusterClient][status]: creating pipe @" + state_jnt_v_pipename)
                os.mkfifo(state_jnt_v_pipename)
            
        self._pipes_created = True

    def _open_pipes(self):

        for i in range(self.cluster_size):

            self.success_pipes_fd[i] = os.open(self.success_pipenames[i],  
                                                os.O_RDONLY | os.O_NONBLOCK)

            # cmds from controllers
            self.cmd_jnt_q_pipes_fd[i] = os.open(self.cmd_jnt_q_pipenames[i], os.O_RDONLY | os.O_NONBLOCK)
            self.cmd_jnt_v_pipes_fd[i] = os.open(self.cmd_jnt_v_pipenames[i], os.O_RDONLY | os.O_NONBLOCK)
            self.cmd_jnt_eff_pipes_fd[i] = os.open(self.cmd_jnt_eff_pipenames[i], os.O_RDONLY | os.O_NONBLOCK)
            
            # state to controllers 
            self.state_root_q_pipe_fd[i] = os.open(self.state_root_q_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_root_v_pipe_fd[i] = os.open(self.state_root_v_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_jnt_q_pipe_fd[i] = os.open(self.state_jnt_q_pipenames[i], 
                                                    os.O_WRONLY)
            self.state_jnt_v_pipe_fd[i] = os.open(self.state_jnt_v_pipenames[i], 
                                                    os.O_WRONLY)
            
        self._pipes_initialized = True

    def _handshake(self):
        
        # THIS RUNS IN A CHILD PROCESS --> we perform the "handshake" with
        # the server: we exchange crucial info which has to be shared between 
        # them

        print("[ControlClusterClient][status][handshake]: opening cluster pipe")
        self.cluster_size_pipe_fd_mp.value = os.open(self.cluster_size_pipe, os.O_WRONLY) # this will block until
        # the server will open it in read mode
        print("[ControlClusterClient][status][handshake]: cluster pipe opened")

        cluster_size_data = struct.pack('i', self.cluster_size)
        print("[ControlClusterClient][status][handshake]: writing to cluster pipe")
        os.write(self.cluster_size_pipe_fd_mp.value, cluster_size_data) # the server is listening -> we send the info we need

        print("[ControlClusterClient][status][handshake]: opening jnt number pipe")
        self.jnt_number_pipe_fd_mp.value = os.open(self.jnt_number_pipe, os.O_RDONLY)
        print("[ControlClusterClient][status][handshake]: reading from jnt number pipe")
        jnt_number_raw = os.read(self.jnt_number_pipe_fd_mp.value, 4)
        self.n_dofs.value = struct.unpack('i', jnt_number_raw)[0]

        import numpy as np
        data = np.zeros((self.n_dofs.value, 1))
        element_size = data.itemsize
        self.jnt_data_size.value = data.shape[0] * data.shape[1] * element_size

        self._is_cluster_ready.value = True # we signal the main process
        # the connection is established

        print("[ControlClusterClient][status][handshake]: connection with ControlCluster server achieved")

    def _trigger_solution(self, 
                        index: int):

        # solver
        trigger_pipes_fd = os.open(self.trigger_pipenames[index],
                                    os.O_WRONLY) # blocking (non-blocking
        # would throw error if nothing has opened the pipe in read mode)
        
        while True: # we keep the process alive

            if self._trigger_solve[index]: # this is set by the parent process

                # Send a signal to perform the solution
         
                os.write(trigger_pipes_fd, b'solve\n')

                self._trigger_solve[index] = False # we will wait for next signal
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

        self._open_pipes()
        print("[ControlClusterClient][status]: pipe opening completed")

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
     
    def solve(self):

        # solve all the TO problems in the control cluster

        if (self._is_cluster_ready.value and self._pipes_initialized):
            
            for i in range(0, self.cluster_size):

                self._trigger_solve[i] = True # we signal the triggger process n.{i} to trigger the solution 
                # of the associated controller

            start_time = time.time()

            # Wait for all operations to complete
            for i in range(self.cluster_size):
                            
                while True:

                    try:

                        response = os.read(self.success_pipes_fd[i], 1024).decode().strip()

                        if response == "success":
                            
                            read_q = os.read(self.cmd_jnt_q_pipes_fd[i], self.jnt_data_size.value)
                            read_v = os.read(self.cmd_jnt_v_pipes_fd[i], self.jnt_data_size.value)
                            read_eff = os.read(self.cmd_jnt_eff_pipes_fd[i], self.jnt_data_size.value)

                            received_q = np.frombuffer(read_q, dtype=np.float32).reshape((1, self.n_dofs.value))
                            received_v = np.frombuffer(read_v, dtype=np.float32).reshape((1, self.n_dofs.value))
                            received_eff = np.frombuffer(read_eff, dtype=np.float32).reshape((1, self.n_dofs.value))

                            print("received q" + str(i) + str(received_q))
                            print("received v" + str(i) + str(received_v))
                            print("received eff" + str(i) + str(received_eff))

                            break

                        else:

                            print("[ControlClusterClient][warning]: received invald response " +  
                                response + " from pipe " + self.success_pipenames[i])

                    except BlockingIOError or SystemError:

                        continue # try again to read

            self.solution_time = time.time() - start_time
            
            self.solution_counter += 1

        if (self._is_cluster_ready.value and (not self._pipes_initialized)):
            
            self._post_initialization() # perform post-initialization steps

        if not self._is_cluster_ready.value:

            print("[ControlClusterClient][status]: waiting connection to ControlCluster server")

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
        
        if self._connection_process.is_alive():
                
            self._connection_process.terminate()  # Forcefully terminate the process
            
            print("[ControlClusterClient][info]: terminating child process " + str(self._connection_process.name))
        
            self._connection_process.join()

        for process in self._trigger_processes:

            if process.is_alive():
                    
                process.terminate()  # Forcefully terminate the process
                
                print("[ControlClusterClient][info]: terminating child process " + str(process.name))
            
                process.join()