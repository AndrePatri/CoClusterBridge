from abc import ABC, abstractmethod

from control_cluster_utils.controllers.rhc import RHChild
from control_cluster_utils.utilities.control_cluster_utils import RobotClusterState, ActionChild

import multiprocess as mp
import os
import struct

from typing import List

class ControlClusterSrvr(ABC):

    def __init__(self, 
                processes_basename: str = "controller", 
                pipe_basepath: str = "/tmp/control_cluster_pipes/"):

        # ciao :D
        #        CR 
        
        self.processes_basename = processes_basename

        self.termination_flag = mp.Value('i', 0)

        self.cluster_size = -1

        self._device = "cpu"

        self._robot_states: RobotClusterState = None

        self._controllers: List[RHChild] = [] # list of controllers (must inherit from
        # RHController)

        self.pipe_basepath = pipe_basepath

        if not os.path.exists(self.pipe_basepath):
            
            print("ControlClusterSrvr: creating pipe folder @ " + self.pipe_basepath)
            os.mkdir(self.pipe_basepath)
    
        # we create several named pipes and store their names

        self.cluster_size_pipe = None
        self.cluster_size_pipe_fd = None
        self.jnt_number_pipe = None
        self.jnt_number_pipe_fd = None

        self._setup_pipes()

        self._processes: List[mp.Process] = [] 

        self._is_cluster_ready = False

        self._controllers_count = 0

        self.solution_time = -1.0

    def _close_processes(self):
    
        # Wait for each process to exit gracefully or terminate forcefully
        
        self.termination_flag.value = 1
        
        for process in self._processes:

            # process.join(timeout=1.0)  # Wait for 5 seconds for each process to exit gracefully

            if process.is_alive():
                
                process.terminate()  # Forcefully terminate the process
            
            print("ControlClusterSrvr: terminating child process " + str(process.name))

    def _clean_pipes(self):

        if os.path.exists(self.cluster_size_pipe):
            
            os.close(self.cluster_size_pipe_fd)
            print("ControlClusterSrvr: closing pipe @" + self.cluster_size_pipe)

        if os.path.exists(self.jnt_number_pipe):
            
            os.close(self.jnt_number_pipe_fd)
            print("ControlClusterSrvr: closing pipe @" + self.jnt_number_pipe)
            
        for i in range(0, self.cluster_size): 

            self._controllers[i].terminate() # closes internal pipes

    def _setup_pipes(self):
        
        self.trigger_pipes = []
        self.success_pipes = []
        self.jnt_q_pipes = []
        self.jnt_v_pipes = []
        self.jnt_eff_pipes = []

        self._connect_to_client() # blocks until a connection with the
        # client is established

    def _connect_to_client(self):
        
        print("ControlClusterSrvr: waiting for connection with the ControlCluster client...")

        # retrieves some important configuration information from the server

        self.cluster_size_pipe = self.pipe_basepath + f"cluster_size.pipe"

        if not os.path.exists(self.cluster_size_pipe):
            
            print("creating pipe @" + self.cluster_size_pipe)
            os.mkfifo(self.cluster_size_pipe)
        
        # open the pipe in read mode with non-blocking option
        self.cluster_size_pipe_fd = os.open(self.cluster_size_pipe, os.O_RDONLY)

        cluster_size_raw = os.read(self.cluster_size_pipe_fd, 4) # we read an integer (32 bits = 4 bytes)

        self.cluster_size = struct.unpack('i', cluster_size_raw)[0]

        self.jnt_number_pipe = self.pipe_basepath + f"jnt_number.pipe"

        if not os.path.exists(self.jnt_number_pipe):
            
            print("creating pipe @" + self.jnt_number_pipe)
            os.mkfifo(self.jnt_number_pipe)

        self.jnt_number_pipe_fd = os.open(self.jnt_number_pipe, os.O_WRONLY)


        for i in range(self.cluster_size):
            
            trigger_pipename = self.pipe_basepath + f"trigger{i}.pipe"
            success_pipename = self.pipe_basepath + f"success{i}.pipe"

            jnt_q_pipename = self.pipe_basepath + f"jnt_q{i}.pipe"
            jnt_v_pipename = self.pipe_basepath + f"jnt_v{i}.pipe"
            jnt_eff_pipename = self.pipe_basepath + f"jnt_eff{i}.pipe"

            if not os.path.exists(trigger_pipename):
                
                print("creating pipe @" + trigger_pipename)
                os.mkfifo(trigger_pipename)
            
            if not os.path.exists(success_pipename):
                
                print("creating pipe @" + success_pipename)
                os.mkfifo(success_pipename)

            if not os.path.exists(jnt_q_pipename):
                
                print("creating pipe @" + jnt_q_pipename)
                os.mkfifo(jnt_q_pipename)

            if not os.path.exists(jnt_v_pipename):
                
                print("creating pipe @" + jnt_v_pipename)
                os.mkfifo(jnt_v_pipename)

            if not os.path.exists(jnt_eff_pipename):
                
                print("creating pipe @" + jnt_eff_pipename)
                os.mkfifo(jnt_eff_pipename)

            self.trigger_pipes.append(trigger_pipename)
            self.success_pipes.append(success_pipename)
            self.jnt_q_pipes.append(jnt_q_pipename)
            self.jnt_v_pipes.append(jnt_v_pipename)
            self.jnt_eff_pipes.append(jnt_eff_pipename)

        print("ControlClusterSrvr: connection to ControlCluster client established.")

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

    def _spawn_processes(self):

        if self._controllers_count == self.cluster_size:
            
            for i in range(0, self.cluster_size):

                process = mp.Process(target=self._controllers[i].solve, 
                                    name = self.processes_basename + str(i))

                self._processes.append(process)

            # we start the processes
            for process in self._processes:

                process.start()

            self._is_cluster_ready = True
                
        else:

            raise Exception("You didn't finish to fill the cluster. Please call the add_controller() method to do so.")

    def _finalize_init(self):

        self.n_dofs = self._controllers[0]._get_ndofs() # we assume all controllers to be for the same robot

        self._robot_states = RobotClusterState(n_dofs = self.n_dofs, 
                                cluster_size = self.cluster_size,
                                device = self._device)
        
        jnt_number_data = struct.pack('i', self.n_dofs)

        os.write(self.jnt_number_pipe_fd, jnt_number_data) # we send this info
        # to the client

    def add_controller(self, controller: RHChild):

        if self._controllers_count < self.cluster_size:

            self._controllers.append(controller)
            
            self._controllers_count += 1

            if self._controllers_count == self.cluster_size:
            
                self._finalize_init()
        
            return True

        if self._controllers_count > self.cluster_size:

            print("Cannot add any more controllers to the cluster. The cluster is full.")

            return False
    
    def start(self):

        self._spawn_processes()

    def terminate(self):

        print("ControlClusterSrvr: terminating cluster")

        self._close_processes() # we also terminate all the child processes

        self._clean_pipes() # we close all the used pipes

    @abstractmethod
    def get(self):

        pass
    
    @abstractmethod
    def set_commands(self,  
                    cluster_cmd: ActionChild):

        pass