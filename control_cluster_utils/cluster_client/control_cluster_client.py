import torch

from abc import ABC

from control_cluster_utils.utilities.control_cluster_defs import RobotClusterState, RobotClusterCmd
from control_cluster_utils.utilities.shared_mem import SharedMemSrvr
from control_cluster_utils.utilities.defs import trigger_flagname

from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

import os
import struct

import time

import numpy as np

import multiprocess as mp

from typing import List

class ControlClusterClient(ABC):

    def __init__(self, 
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            jnt_names: List[str],
            backend = "torch", 
            device = torch.device("cpu"), 
            np_array_dtype = np.float32, 
            verbose = False, 
            debug = False):

        self._verbose = verbose

        self._debug = debug

        self.np_dtype = np_array_dtype
        data_aux = np.zeros((1, 1), dtype=self.np_dtype)
        self.np_array_itemsize = data_aux.itemsize

        if self.np_dtype == np.float64:
            self.torch_dtype = torch.float64
        if self.np_dtype == np.float32:
            self.torch_dtype = torch.float32

        self.jnt_names = jnt_names

        self.n_dofs = len(self.jnt_names)
        self.jnt_data_size = np.zeros((self.n_dofs, 1),
                                    dtype=self.np_dtype).nbytes
        
        self.cluster_size = cluster_size
        
        self.jnt_names = jnt_names

        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs

        self._backend = backend
        self._device = device

        self.robot_states = RobotClusterState(self.n_dofs, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # from robot to controllers

        self.controllers_cmds = None # we wait to know the size of the additional data
        # from the control server
        
        self._was_cluster_ready = False
        self.is_cluster_ready = mp.Value('b', False)
        self.add_data_length = mp.Value('i', 0)
        self._is_first_control_step = False

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"
    
        self.pipes_manager = NamedPipesHandler()
        self.pipes_manager.create_buildpipes()

        self.solution_time = -1.0
        self.n_sim_step_per_cntrl = -1
        self.solution_counter = 0
        self._compute_n_control_actions()

        self._init_shared_flags() # initializes shared memory used for 
        # communication between the client and server

        self._spawn_processes() # we launch all the child processes

    def _spawn_processes(self):
        
        # we spawn the heartbeat() to another process, 
        # so that it's not blocking wrt the simulator

        self._connection_process = mp.Process(target=self._heartbeat, 
                                name = "ControlClusterClient_heartbeat")
        self._connection_process.start()

        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _heartbeat process")

    def _heartbeat(self):
        
        # THIS RUNS IN A CHILD PROCESS --> we perform the "heartbeat" with
        # the server: we exchange crucial info which has to be shared between 
        # them
        
        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": waiting for heartbeat with the ControlCluster server...")

        # retrieves/sends some important configuration information from the server

        # cluster size TO cluster server
        self.pipes_manager.open_pipes(["cluster_size"], 
                                    mode=OMode["O_WRONLY"])
        cluster_size_data = struct.pack('i', self.cluster_size)
        os.write(self.pipes_manager.pipes_fd["cluster_size"], cluster_size_data) # the server is listening -> we send the info we need

        # robot joint number FROM server
        self.pipes_manager.open_pipes(selector=["jnt_number_srvr"], 
                                mode=OMode["O_RDONLY"])
        jnt_number_raw = os.read(self.pipes_manager.pipes_fd["jnt_number_srvr"], DSize["int"])
        n_dofs_srvr_side = struct.unpack('i', jnt_number_raw)[0]

        if len(self.jnt_names) != n_dofs_srvr_side:

            exception = f"[{self.__class__.__name__}]" + f"{self.exception}" + \
                f": the received number of robot joints from cluster server {n_dofs_srvr_side} " + \
                f"does not match the client-side value of {len(self.jnt_names)}"
            
            raise Exception(exception)

        self.pipes_manager.open_pipes(selector=["add_data_length"], 
                                mode=OMode["O_RDONLY"])
        add_data_length_raw = os.read(self.pipes_manager.pipes_fd["add_data_length"], DSize["int"])
        self.add_data_length.value = struct.unpack('i', add_data_length_raw)[0]

        # additional data from solver length

        # client-side robot joints name TO cluster server
        # we first encode the list
        jnt_names_client_string = "\n".join(self.jnt_names)  # convert list to a newline-separated string
        jnt_names_client_raw = jnt_names_client_string.encode('utf-8')
        n_bytes_jnt_names = len(jnt_names_client_raw)

        self.pipes_manager.open_pipes(["n_bytes_jnt_names_client"], 
                                    mode=OMode["O_WRONLY"])
        n_bytes_jnt_names_raw = struct.pack('i', n_bytes_jnt_names)
        os.write(self.pipes_manager.pipes_fd["n_bytes_jnt_names_client"], n_bytes_jnt_names_raw)

        self.pipes_manager.open_pipes(["jnt_names_client"], 
                                    mode=OMode["O_WRONLY"])
        
        os.write(self.pipes_manager.pipes_fd["jnt_names_client"], 
                jnt_names_client_raw)

        self.is_cluster_ready.value = True # we signal the main process
        # the connection is established

        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": friendship with ControlCluster server established.")

    def _init_shared_flags(self):

        dtype = torch.bool # using a boolean type shared data, 
        # exposes low-latency boolean writing and reading methods

        self.trigger_flags = SharedMemSrvr(self.cluster_size, 1, 
                                trigger_flagname(), 
                                dtype=dtype) 
    
    def _trigger_solution(self):

        if self.is_cluster_ready.value: 

            self.trigger_flags.reset_bool(True) # sets all flags

    def _solved(self):

        if self.is_cluster_ready.value: 
            
            while not self.trigger_flags.none(): # far too much CPU intensive?

                continue

            return True
            
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
    
    def _finalize_init(self):
        
        # things to be done when everything is set but before starting to solve

        self.controllers_cmds = RobotClusterCmd(self.n_dofs, 
                                            cluster_size=self.cluster_size,
                                            add_data_size = self.add_data_length.value, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # now that we know add_data_size
        # we can initialize the control commands
    
    def cluster_ready(self):

        return self._was_cluster_ready and self.is_cluster_ready.value
    
    def is_first_control_step(self):

        return self._is_first_control_step
    
    def solve(self):

        # solve all the TO problems in the control cluster

        is_cluster_ready = self.is_cluster_ready.value

        if not is_cluster_ready:

            if self._verbose: 

                print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": waiting connection to ControlCluster server")

        if self._is_first_control_step:
                
                self._is_first_control_step = False
                
        if (not self._was_cluster_ready) and is_cluster_ready:
            
            # first time the cluster is ready 

            self._finalize_init() # we perform the final initializations

            self._was_cluster_ready = True

            self._is_first_control_step = True

        if (is_cluster_ready):
            
            if self._debug:

                start_time = time.perf_counter() # we profile the whole solution pipeline
            
            self.robot_states.synch() # updates shared tensor on CPU with data from states on GPU

            self._trigger_solution() # triggers solution of all controllers in the cluster 

            # we wait for all controllers to finish      
            solved = self._solved() # this is blocking
                
            # at this point all controllers are done -> we synchronize the control commands on GPU
            # with the ones written by each controller on CPU
            self.controllers_cmds.synch()

            self.solution_counter += 1

            if self._debug:

                self.solution_time = time.perf_counter() - start_time # we profile the whole solution pipeline
        
    def close(self):

        self.__del__()
    
    def _close_process(self, 
                    process):
        
        if process.is_alive():
            
            process.terminate()  # Forcefully terminate the process
                    
            print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": terminating child process " + str(process.name))
        
            process.join()
        
    def __del__(self):
        
        if self._connection_process is not None:
            
            self._close_process(self._connection_process)