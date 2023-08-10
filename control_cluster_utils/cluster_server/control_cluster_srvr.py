from abc import ABC

from control_cluster_utils.controllers.rhc import RHChild

from control_cluster_utils.utilities.control_cluster_defs import HanshakeDataCntrlSrvr
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

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        self.processes_basename = processes_basename

        self.cluster_size = -1

        self._device = "cpu"

        self._controllers: List[RHChild] = [] # list of controllers (must inherit from
        # RHController)

        self.handshake_srvr = HanshakeDataCntrlSrvr()
        self.handshake_srvr.handshake()
        self.cluster_size = self.handshake_srvr.cluster_size.tensor_view[0, 0].item()

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

        self.handshake_srvr.finalize_init(self._controllers[0].add_data_lenght)

        self.client_side_jnt_names = self.handshake_srvr.jnt_names_client.read()

        for i in range(0, self.cluster_size):

            # we assign the client-side joint names to each controller (used for mapping purposes)
            self._controllers[i].assign_client_side_jnt_names(self.client_side_jnt_names)

            self._controllers[i].create_jnt_maps()

            self._controllers[i].init_states() # initializes states

        self._check_jnt_names_compatibility() 
    
        print(f"[{self.__class__.__name__}]" + f"{self.status}" + ": final initialization steps completed.")

    def _check_jnt_names_compatibility(self):

        set_srvr = set(self.server_side_jnt_names)
        set_client  = set(self.client_side_jnt_names)
        
        print("####################")
        print(str(self.server_side_jnt_names))
        print(str(self.client_side_jnt_names))
        print("#####################")
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