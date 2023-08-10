from abc import ABC, abstractmethod

from control_cluster_utils.controllers.rhc import RHChild
from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

import os
import struct

from typing import List

import multiprocess as mp

from typing import TypeVar

ClusterSrvrChild = TypeVar('ClusterSrvrChild', bound='ControlClusterSrvr')

class ControlClusterSrvr(ABC):

    def __init__(self, 
                processes_basename: str = "controller"):

        # ciao :D
        #        CR 
        
        self.pipes_manager = NamedPipesHandler() # object to better handle 
        self.pipes_manager.create_buildpipes()

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        self.processes_basename = processes_basename

        self.cluster_size = -1

        self._device = "cpu"

        self._controllers: List[RHChild] = [] # list of controllers (must inherit from
        # RHController)

        self._handshake() 

        self._processes: List[mp.Process] = [] 

        self._is_cluster_ready = False

        self._controllers_count = 0

        self.solution_time = -1.0

        self.client_side_jnt_names = []
        self.server_side_jnt_names = []

    def _close_processes(self):
    
        # Wait for each process to exit gracefully or terminate forcefully
                
        for process in self._processes:

            process.join(timeout=0.2)  # Wait for 5 seconds for each process to exit gracefully

            if process.is_alive():
                
                process.terminate()  # Forcefully terminate the process
            
            print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": terminating child process " + str(process.name))

    def _clean_pipes(self):

        self.pipes_manager.close_pipes(selector=["cluster_size", "jnt_number"])
        
    def _handshake(self):
        
        print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": waiting for handshake with the ControlCluster client...")

        # retrieves some important configuration information from the server

        # cluster size
        self.pipes_manager.open_pipes(["cluster_size"], 
                                    mode=OMode["O_RDONLY"])
        cluster_size_raw = os.read(self.pipes_manager.pipes_fd["cluster_size"], DSize["int"])
        # this will block until we get the info from the client
        self.cluster_size = struct.unpack('i', cluster_size_raw)[0]
        
        self.pipes_manager.create_runtime_pipes(self.cluster_size) # we create the remaining runtime pipes, now that we now the cluster size

        print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": handshake with ControlCluster client completed.")

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

            raise Exception(f"[{self.__class__.__name__}]" + f"{self.exception}" + "You didn't finish to fill the cluster. Please call the add_controller() method to do so.")

    def _finalize_init(self):

        # steps to be performed after the controllers are fully initialized 

        print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": performing final initialization steps...")

        self.n_dofs = self._controllers[0]._get_ndofs() # we assume all controllers to be for the same robot
        self.server_side_jnt_names = self._controllers[0]._server_side_jnt_names
        
        # we send the joint number to the client 
        jnt_number_srvr_data = struct.pack('i', self.n_dofs)
        self.pipes_manager.open_pipes(selector=["jnt_number_srvr"], 
                                mode=OMode["O_WRONLY"])
        os.write(self.pipes_manager.pipes_fd["jnt_number_srvr"], jnt_number_srvr_data) # we send this info
        # to the client, which is now guaranteed to be listening on the pipe

        # we send the add_data_length to the client 
        add_data_lenght_data = struct.pack('i', self._controllers[0].add_data_lenght)
        self.pipes_manager.open_pipes(selector=["add_data_length"], 
                                mode=OMode["O_WRONLY"])
        os.write(self.pipes_manager.pipes_fd["add_data_length"], add_data_lenght_data) # we send this info
        # to the client, which is now guaranteed to be listening on the pipe
        
        # client-side joint names (e.g. coming from the simulator)

        self.pipes_manager.open_pipes(["n_bytes_jnt_names_client"], 
                                    mode=OMode["O_RDONLY"])
        n_bytes_jnt_names_client_raw = os.read(self.pipes_manager.pipes_fd["n_bytes_jnt_names_client"], DSize["int"])
        # this will block until we get the info from the client
        n_bytes_jnt_names_client = struct.unpack('i', n_bytes_jnt_names_client_raw)[0]
     
        self.pipes_manager.open_pipes(["jnt_names_client"], 
                                    mode=OMode["O_RDONLY"])
        jnt_names_client_raw = os.read(self.pipes_manager.pipes_fd["jnt_names_client"], 
                                n_bytes_jnt_names_client) # reads all the data available
        jnt_names_client_str = jnt_names_client_raw.decode('utf-8')
        jnt_names_client_list_raw = jnt_names_client_str.split("\n") # we suppose each joint name to be separated by a newline character
        self.client_side_jnt_names = [s for s in jnt_names_client_list_raw if s] # remove any empty strings in case there's an extra newline at the end

        for i in range(0, self.cluster_size):

            # we assign the client-side joint names to each controller (used for mapping purposes)
            self._controllers[i].assign_client_side_jnt_names(self.client_side_jnt_names)

            self._controllers[i].create_jnt_maps() # this is called anyway inside the solve, but this way
            # we save a bit of time when spawning the processes

            self._controllers[i].init_states() # initializes states

        self._check_jnt_names_compatibility() 
    
        print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": final initialization steps completed.")

    def _check_jnt_names_compatibility(self):

        set_srvr = set(self.server_side_jnt_names)
        set_client  = set(self.client_side_jnt_names)
        
        if not set_srvr == set_client:

            exception = f"[{self.__class__.__name__}]" + f"{self.exception}" + ": server-side and client-side joint names do not match!"

            raise Exception(exception)
    
    def add_controller(self, controller: RHChild):

        if self._controllers_count < self.cluster_size:

            self._controllers.append(controller)
            
            self._controllers_count += 1

            if self._controllers_count == self.cluster_size:
            
                self._finalize_init()
        
            return True

        if self._controllers_count > self.cluster_size:

            print(f"[{self.__class__.__name__}]" + f"[{self.warning}]" + ": cannot add any more controllers to the cluster. The cluster is full.")

            return False
    
    def start(self):

        self._spawn_processes()

    def terminate(self):

        print(f"[{self.__class__.__name__}]" + f"[{self.info}]" + ": terminating cluster")

        self._close_processes() # we also terminate all the child processes

        self._clean_pipes() # we close all the used pipes