import torch

from abc import ABC, abstractmethod

from control_cluster_utils.utilities.control_cluster_utils import RobotClusterState, ActionChild

import os
import struct

import time

import numpy as np

class ControlClusterClient(ABC):

    def __init__(self, 
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            backend: str = "torch", 
            device: str = "cpu"):
        
        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs

        self.n_dofs = None
        self.jnt_data_size = -1
        self.cluster_size = cluster_size

        self._backend = backend

        self._device = device

        self._robot_states: RobotClusterState = None

        self.pipe_basepath = "/tmp/control_cluster_pipes/"
        
        self.cluster_size_pipe = None
        self.cluster_size_pipe_fd = None
        self.jnt_number_pipe = None
        self.jnt_number_pipe_fd = None
        # we create several named pipes and store their names
        self.trigger_pipes = []
        self.success_pipes = []
        self.jnt_q_pipes = []
        self.jnt_v_pipes = []
        self.jnt_eff_pipes = []

        self.trigger_pipes_fd = []
        self.success_pipes_fd = []
        self.jnt_q_pipes_fd = []
        self.jnt_v_pipes_fd = []
        self.jnt_eff_pipes_fd = []

        self._is_cluster_ready = False

        self.solution_time = -1.0
        self.n_sim_step_per_cntrl = -1
        self.solution_counter = 0
        self._compute_n_control_actions()

        self._setup_pipes()

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

    def _setup_pipes(self):
        
        if not os.path.exists(self.pipe_basepath):
            
            print("ControlClusterClient: creating pipes directory @ " + self.pipe_basepath)
            os.mkdir(self.pipe_basepath)

        self.cluster_size_pipe = self.pipe_basepath + f"cluster_size.pipe"
        if not os.path.exists(self.cluster_size_pipe):
            
            print("ControlClusterClient: creating pipe @ " + self.cluster_size_pipe)
            os.mkfifo(self.cluster_size_pipe)

        self.jnt_number_pipe = self.pipe_basepath + f"jnt_number.pipe"

        if not os.path.exists(self.jnt_number_pipe):
            
            print("ControlClusterClient: creating pipe @" + self.jnt_number_pipe)
            os.mkfifo(self.jnt_number_pipe)

        print("ControlClusterClient: waiting connection to ControlCluster server")
        self._connect() # wait for the server to connect
        print("ControlClusterClient: connection with ControlCluster server achieved")

        self._is_cluster_ready = True

    def _connect(self):
               
        self.cluster_size_pipe_fd = os.open(self.cluster_size_pipe, os.O_WRONLY)

        cluster_size_data = struct.pack('i', self.cluster_size)

        os.write(self.cluster_size_pipe_fd, cluster_size_data)

        self.jnt_number_pipe_fd = os.open(self.jnt_number_pipe, os.O_RDONLY)

        for i in range(self.cluster_size):
            
            trigger_pipename = self.pipe_basepath + f"trigger{i}.pipe"
            success_pipename = self.pipe_basepath + f"success{i}.pipe"

            jnt_q_pipename = self.pipe_basepath + f"jnt_q{i}.pipe"
            jnt_v_pipename = self.pipe_basepath + f"jnt_v{i}.pipe"
            jnt_eff_pipename = self.pipe_basepath + f"jnt_eff{i}.pipe"

            self.trigger_pipes.append(trigger_pipename)
            self.success_pipes.append(success_pipename)
            self.jnt_q_pipes.append(jnt_q_pipename)
            self.jnt_v_pipes.append(jnt_v_pipename)
            self.jnt_eff_pipes.append(jnt_eff_pipename)

            # we wait for pipes creation from the server

            while not os.path.exists(trigger_pipename): 

                continue 

            while not os.path.exists(trigger_pipename):
                    
                continue
            
            while not os.path.exists(success_pipename):
                
                continue

            while not os.path.exists(jnt_q_pipename):
                
                continue

            while not os.path.exists(jnt_v_pipename):
                
                continue

            while not os.path.exists(jnt_eff_pipename):
                
                continue


            self.success_pipes_fd.append(os.open(success_pipename,  os.O_RDONLY | os.O_NONBLOCK))
            self.trigger_pipes_fd.append(os.open(trigger_pipename, os.O_WRONLY)) # this will block until
            # something opens the pipe in read mode

            self.jnt_q_pipes_fd.append(os.open(jnt_q_pipename, os.O_RDONLY | os.O_NONBLOCK))
            self.jnt_v_pipes_fd.append(os.open(jnt_v_pipename, os.O_RDONLY | os.O_NONBLOCK))
            self.jnt_eff_pipes_fd.append(os.open(jnt_eff_pipename, os.O_RDONLY | os.O_NONBLOCK))

        jnt_number_raw = os.read(self.jnt_number_pipe_fd, 4)
        self.n_dofs = struct.unpack('i', jnt_number_raw)[0]

        import numpy as np
        data = np.zeros((self.n_dofs, 1))
        element_size = data.itemsize
        self.jnt_data_size = data.shape[0] * data.shape[1] * element_size
    
    def _check_state_size(self, 
                        cluster_state: RobotClusterState):

        if cluster_state.n_dofs != self.n_dofs:

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

        if (self._is_cluster_ready):
            
            # Send a signal to each process to perform the operation
            for i in range(self.cluster_size):

                # print("sending solution signal to controller n." + str(i))

                os.write(self.trigger_pipes_fd[i], b'solve\n')

            start_time = time.time()

            print("Reading success signal")

            # Wait for all operations to complete
            for i in range(self.cluster_size):
            
                # print("waiting for solution from controller n." + str(i))
                
                while True:

                    try:

                        success = os.read(self.success_pipes_fd[i], 1024).decode().strip()

                        if success == "success":
                            
                            read_q = os.read(self.jnt_q_pipes_fd[i], self.jnt_data_size)
                            read_v = os.read(self.jnt_v_pipes_fd[i], self.jnt_data_size)
                            read_eff = os.read(self.jnt_eff_pipes_fd[i], self.jnt_data_size)

                            received_q = np.frombuffer(read_q, dtype=np.float32).reshape((1, self.n_dofs))
                            received_v = np.frombuffer(read_v, dtype=np.float32).reshape((1, self.n_dofs))
                            received_eff = np.frombuffer(read_eff, dtype=np.float32).reshape((1, self.n_dofs))

                            print("received q" + str(i) + str(received_q))
                            print("received v" + str(i) + str(received_v))
                            print("received eff" + str(i) + str(received_eff))

                            break

                    except BlockingIOError or SystemError:

                        continue # try again to read

            print("Read succes signal")

            self.solution_time = time.time() - start_time
            
            self.solution_counter += 1

        else:

            raise Exception("The cluster client is not initialized properly.")

    @abstractmethod
    def get(self):

        pass
    
    @abstractmethod
    def set_commands(self,  
                    cluster_cmd: ActionChild):

        pass
