import torch

from abc import ABC

from control_cluster_utils.utilities.control_cluster_defs import RobotClusterState, RobotClusterCmd
from control_cluster_utils.utilities.control_cluster_defs import HanshakeDataCntrlClient
from control_cluster_utils.utilities.control_cluster_defs import RhcClusterTaskRefs

from control_cluster_utils.utilities.shared_mem import SharedMemSrvr
from control_cluster_utils.utilities.defs import trigger_flagname, launch_controllers_flagname

import time

import numpy as np

import threading

from typing import List

from perf_sleep.pyperfsleep import PerfSleep

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

        self.perf_timer = PerfSleep()

        self._verbose = verbose

        self._terminate = False

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

        # shared mem objects
        self.handshake_manager = None
        self._handshake_thread = None
        self.robot_states = None
        self.controllers_cmds = None
        self.trigger_flags = None
        self.launch_controllers = None
        self.rhc_task_refs = None

        # flags
        self._was_cluster_ready = False
        self.is_cluster_ready = False
        self._is_first_control_step = False

        # message handling
        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"

        # other data
        self.add_data_length = 0

        self.solution_time = -1.0
        self.solution_counter = 0
        self.n_sim_step_per_cntrl = -1

        # performs some initialization steps
        self._setup()
        
    def _setup(self):

        self._compute_n_control_actions() # necessary ti apply control input only at 
        # a specific rate

        self._init_shared_mem() # initializes shared memory used for 
        # communication between the client and server
        self._start_shared_mem() # starts memory servers and clients

    def _start_shared_mem(self):

        self.robot_states.start()

        self.trigger_flags.start()

        self.launch_controllers.start()
        self.launch_controllers.reset_bool(False)

        self._spawn_handshake() # we launch all the child processes

    def _spawn_handshake(self):
        
        # we spawn the heartbeat() to another process, 
        # so that it's not blocking wrt the simulator

        self._handshake_thread =  threading.Thread(target=self.handshake_manager.start, 
                                args=(self.cluster_size, self.jnt_names, ), 
                                kwargs={})
        
        self._handshake_thread.start()

        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + \
            ": spawned _heartbeat thread")

    def _init_shared_mem(self):
        
        self.robot_states = RobotClusterState(self.n_dofs, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # from robot to controllers

        self.handshake_manager = HanshakeDataCntrlClient(self.n_dofs) # handles handshake process
        # between client and server

        dtype = torch.bool # using a boolean type shared data, 
        # exposes low-latency boolean writing and reading methods

        self.trigger_flags = SharedMemSrvr(self.cluster_size, 1, 
                                trigger_flagname(), 
                                dtype=dtype) 
        
        self.launch_controllers = SharedMemSrvr(1, 1, 
                                launch_controllers_flagname(), 
                                dtype=dtype) 

    def _trigger_solution(self):

        self.trigger_flags.reset_bool(True) # sets all flags

    def _solved(self):

        solved = False
            
        while not self.trigger_flags.none(): # far too much CPU intensive?

            if (not self._terminate) and \
                (self.trigger_flags.get_clients_count() == self.cluster_size):
                
                self.perf_timer.clock_sleep(1000) # nanoseconds (but this
                # # accuracy cannot be reached on a non-rt system)
                # # on a modern laptop, this sleeps for about 5e-5s

                continue
            
            else:
                
                solved = False

                break
        
        solved = True

        return solved
        
    def _compute_n_control_actions(self):

        if self.cluster_dt < self.control_dt:

            print(f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + \
                ": cluster_dt has to be >= control_dt")

            self.n_sim_step_per_cntrl = 1
        
        else:
            
            self.n_sim_step_per_cntrl = round(self.cluster_dt / self.control_dt)
            self.cluster_dt = self.control_dt * self.n_sim_step_per_cntrl

        message = f"[{self.__class__.__name__}]"  + f"[{self.info}]" + \
                ": the cluster controllers will run at a rate of " + \
                str(1.0 / self.cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self.control_dt) + "Hz.\n" + \
                "Number of sim steps per control step: " + str(self.n_sim_step_per_cntrl)

        print(message)
    
    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index the current simulation loop number (0-based)

        return (control_index + 1) % self.n_sim_step_per_cntrl == 0
    
    def _finalize_init(self):
        
        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + \
                    ": connecting to server...")
        
        # things to be done when everything is set but before starting to solve

        add_data_length_from_server = self.handshake_manager.add_data_length.tensor_view[0, 0].item()
        n_contacts_from_server = self.handshake_manager.n_contacts.tensor_view[0, 0].item()

        self.controllers_cmds = RobotClusterCmd(self.n_dofs, 
                                            cluster_size=self.cluster_size,
                                            add_data_size = add_data_length_from_server, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # now that we know add_data_size
        # we can initialize the control commands
        self.controllers_cmds.start()

        self.rhc_task_refs = RhcClusterTaskRefs(n_contacts=n_contacts_from_server, 
                                    cluster_size=self.cluster_size, 
                                    device=self._device, 
                                    backend=self._backend, 
                                    dtype=self.torch_dtype)
        self.rhc_task_refs.start()
        
        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + \
                    ": connection achieved.")
        
    def cluster_ready(self):

        return self._was_cluster_ready and self.handshake_manager.handshake_done
    
    def is_first_control_step(self):

        return self._is_first_control_step
    
    def solve(self):

        # solve all the TO problems in the control cluster

        handshake_done = self.handshake_manager.handshake_done

        if not handshake_done or (self.trigger_flags.get_clients_count() != self.cluster_size):

            if self._verbose: 

                print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + \
                    ": waiting connection to ControlCluster server")

        if self._is_first_control_step:
                
            self._is_first_control_step = False
                
        if (not self._was_cluster_ready) and handshake_done:
            
            # first time the cluster is ready 

            self._finalize_init() # we perform the final initializations

            self._was_cluster_ready = True

            self._is_first_control_step = True

            self.is_cluster_ready = True

        if self.is_cluster_ready:
            
            if self._debug:

                start_time = time.perf_counter() # we profile the whole solution pipeline
            
            self.robot_states.synch() # updates shared tensor on CPU with data from states on GPU

            if self.launch_controllers.all():
                
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
        
        if not self._terminate:
            
            self._terminate = True
            
            if self.robot_states is not None:
                
                self.robot_states.terminate()

            if self.trigger_flags is not None:
                
                self.trigger_flags.terminate()
            
            if self.launch_controllers is not None:

                self.launch_controllers.terminate()

            if self.controllers_cmds is not None:
                
                self.controllers_cmds.terminate()

            if self.rhc_task_refs is not None:

                self.rhc_task_refs.terminate()

        self._close_handshake()

    def _close_handshake(self):

        if self.handshake_manager is not None:
                
                self.handshake_manager.terminate() # will close/detach all shared memory 
                
        if self._handshake_thread is not None:
                
            self._close_thread(self._handshake_thread) # we first wait for thread to exit, if still alive

    def _close_thread(self, 
                    thread):
        
        if thread.is_alive():
                        
            print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + \
                ": terminating child thread " + str(thread.name))
        
            thread.join() # wait for thread to join

    def __del__(self):
                
        self.close()