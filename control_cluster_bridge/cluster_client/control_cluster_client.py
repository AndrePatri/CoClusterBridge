# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
import torch

from abc import ABC

from control_cluster_bridge.utilities.control_cluster_defs import RhcClusterTaskRefs

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState 
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling
from control_cluster_bridge.utilities.shared_data.handshaking import HandShaker

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType

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
            n_contact_sensors: int = -1,
            contact_linknames: List[str] = None,
            backend = "torch", 
            device = torch.device("cpu"), 
            np_array_dtype = np.float32, 
            verbose = False, 
            debug = False, 
            namespace = ""):

        self.namespace = namespace
        
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
        self.using_gpu = False
        if self._device == torch.device("cuda"):
            self.using_gpu = True

        # shared mem objects
        self._handshaker = None
        self._handshake_thread2 = None
        
        self.robot_states = None
        self.rhc_cmds = None

        self.rhc_task_refs = None
        # self.shared_cluster_info = None

        self.rhc_status = None
        self.cluster_stats = None 
        
        # flags
        self._was_cluster_ready = False
        self.is_cluster_ready = False
        self._is_first_control_step = False

        self.controllers_were_active = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool)

        # other data
        self.add_data_length = 0
        self.n_contact_sensors = n_contact_sensors
        self.contact_linknames = contact_linknames

        self.solution_time = np.nan
        self.start_time = np.nan
        self.solution_counter = 0
        self.n_sim_step_per_cntrl = -1

        self.trigger_counter = 0
        self.wait_for_sol_counter = 0
        self.n_steps_prints = 100
        # performs some initialization steps
        self._setup()

    def __del__(self):
                
        self.close()

    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index is the current simulation loop number (0-based)
        
        # returns true if this is a control "instant"
        
        return (control_index + 1) % self.n_sim_step_per_cntrl == 0
    
    def cluster_ready(self):

        return self.is_cluster_ready
    
    def is_first_control_step(self):

        return self._is_first_control_step
    
    def _sporadic_log(self,
                calling_methd: str,
                msg: str,
                logtype: LogType = LogType.INFO):

        if self._verbose and \
            (self.trigger_counter+1) % self.n_steps_prints == 0: 
            
            Journal.log(self.__class__.__name__,
                calling_methd,
                msg,
                logtype,
                throw_when_excep = True)
                
    def trigger_solution(self):

        # performs checks and triggers cluster solution

        self.rhc_status.controllers_counter.synch_all(wait = True,
                                                    read = True)
        n_clients = self.rhc_status.controllers_counter.torch_view[0, 0]
        
        if n_clients == 0:
            
            self._sporadic_log(calling_methd="trigger_solution",
                        msg = "waiting connection to ControlCluster server")

        if n_clients < self.cluster_size and \
                n_clients > 0 and \
                (self.trigger_counter+1) % self.n_steps_prints == 0:
                                            
            self._sporadic_log(calling_methd="trigger_solution",
                    msg = f"Not all clients are connected yet ({n_clients}/{self.cluster_size}).",
                    logtype=LogType.WARN)

        if n_clients > self.cluster_size:
            
            msg = f"More than {self.cluster_size} controllers registered " + \
                f"(total of {n_clients})." + \
                ". It's very likely a memory leak on the shared memory layer occurred." + \
                " You might need to reboot the system to clean the dangling memory."

            self._sporadic_log(calling_methd="trigger_solution",
                        msg = msg,
                        logtype=LogType.EXCEP)
            
        if self._is_first_control_step:
                
            self._is_first_control_step = False
                
        if (not self._was_cluster_ready):
            
            # first time the cluster is ready 

            self._finalize_init() # we perform the final initializations

            self._was_cluster_ready = True

            self._is_first_control_step = True

            self.is_cluster_ready = True

        # solve all the TO problems in the control cluster

        if self._debug and self.is_cluster_ready:
            
            # we profile the whole solution pipeline
            # [
            # robot and contact state update/synchronization, 
            # solution triggering, 
            # commands reading/synchronization, 
            # ]
            # note that this profiling can be influenced by other operations
            # done between the call to trigger_solution() and wait_for_solution()
            # for example if simulation stepping is carried out in between and is slower
            # than the cluster solution, this will increase the profiled time as if the 
            # cluster was slower

            self.start_time = time.perf_counter() 

        if self.robot_states is not None:
            
            if self.using_gpu:

                # updates shared tensor on CPU with latest data from states on GPU
                # and writes to shared mem
                self.robot_states.synch_mirror(from_gpu=True)
            
            else:
                
                # only write to shared mem (gpu mirror is not used)
                self.robot_states.synch_to_shared_mem()

        if self.is_cluster_ready:
                            
            self._trigger_solution() # actually triggers solution of all controllers in the cluster 
            # which are active using the latest available state
            
            if not self.rhc_status.activation_state.torch_view.all():
                
                msg = f"Controllers waiting to be activated... (" + \
                    f"{self.rhc_status.activation_state.torch_view.sum().item()}/{self.cluster_size})"
                
                self._sporadic_log(calling_methd="trigger_solution",
                            msg = msg,
                            logtype=LogType.INFO)

        self.trigger_counter +=1

    def _on_failure(self):

        # checks failure status
        self.rhc_status.fails.synch_all(read=True, 
                                    wait=True)
        
        # writes failures to reset flags
        self.rhc_status.resets.write_wait(self.rhc_status.fails.torch_view,
                                    0, 0)
    
    def wait_for_solution(self):

        if self.is_cluster_ready:
            
            # we wait for all controllers to finish      
            done = self._wait_for_solution() # this is blocking if at least a controller
            # is active
            
            # we handle controller fails, if any
            self._on_failure()

            # at this point all controllers are done -> we synchronize the control commands on GPU
            # with the ones written by each controller on CPU

            if done:

                if self.using_gpu:
                    
                    # first reads cmds from shared memory and then synchs gpu mirror with the read data
                    self.rhc_cmds.synch_mirror(from_gpu=False) 

                else:
                    
                    # only read cmds from shared mem
                    self.rhc_cmds.synch_from_shared_mem()

                self.solution_counter += 1

        if self._debug and self.is_cluster_ready:
            
            # this is updated even if the cluster is not active (values do not make sense
            # in that case)

            self.solution_time = time.perf_counter() - self.start_time # we profile the whole solution pipeline
            
            # self.shared_cluster_info.update(solve_time=self.solution_time, 
            #                             controllers_up = self.controllers_active) # we update the shared info

            # self.cluster_stats.write_info(dyn_info_name=["cluster_sol_time", 
            #                             "cluster_rt_factor",
            #                             "cluster_ready",
            #                             "cluster_state_update_dt"],
            #             val=[self.solution_time,
            #                 self.cluster_dt/self.solution_time,
            #                 self.cluster_ready(),
            #                 np.nan])
    
        self.controllers_were_active[:, :] = self.rhc_status.activation_state.torch_view

        self.wait_for_sol_counter +=1

    def close(self):
        
        if not self._terminate:
            
            self._terminate = True
            
            if self.robot_states is not None:
                
                self.robot_states.close()

            if self.rhc_cmds is not None:
                
                self.rhc_cmds.close()

            if self.rhc_task_refs is not None:

                self.rhc_task_refs.terminate()

            if self.rhc_status is not None:

                self.rhc_status.close()
            
            if self.cluster_stats is not None:

                self.cluster_stats.close()

        # self._close_handshake()

    def _close_handshake(self):
        
        if self._handshaker is not None:

            self._handshaker.close()

        if self._handshake_thread2 is not None:

            self._close_thread(self._handshake_thread2)

    def _close_thread(self, 
                    thread):
        
        if thread.is_alive():
            
            Journal.log(self.__class__.__name__,
                    "_close_thread",
                    "Terminating child thread " + str(thread.name),
                    LogType.INFO,
                    throw_when_excep = False)
        
            thread.join() # wait for thread to join

    def _setup(self):

        self._compute_n_control_actions() # necessary ti apply control input only at 
        # a specific rate

        self._init_shared_mem() # initializes shared memory used for 
        # communication between the client and server
        self._start_shared_mem() # starts memory servers and clients

    def _start_shared_mem(self):

        self.robot_states.run()

        self.rhc_cmds.run()

        # self.shared_cluster_info.start()

        self.rhc_status.run()

        # self._spawn_handshake() # we launch all the child processes

    def _spawn_handshake(self):
        
        # we spawn the heartbeat() to another process, 
        # so that it's not blocking wrt the simulator

        self._handshake_thread2 = threading.Thread(target=self._handshaker.run, 
                                args=(), 
                                kwargs={})
        
        self._handshake_thread2.start()

        Journal.log(self.__class__.__name__,
                    "_spawn_handshake",
                    "Spawned _heartbeat thread",
                    LogType.INFO,
                    throw_when_excep = False)

    def _init_shared_mem(self):
                
        self.robot_states = RobotState(namespace=self.namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self.n_contact_sensors,
                                jnt_names=self.jnt_names,
                                contact_names=self.contact_linknames,
                                with_gpu_mirror=True,
                                force_reconnection=False,
                                verbose=True,
                                vlevel=VLevel.V2,
                                safe=False)

        self.rhc_cmds = RhcCmds(namespace=self.namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self.n_contact_sensors,
                                jnt_names=self.jnt_names,
                                contact_names=self.contact_linknames,
                                with_gpu_mirror=True,
                                force_reconnection=False,
                                verbose=True,
                                vlevel=VLevel.V2,
                                safe=False)
                                            
        # between client and server

        dtype = torch.bool # using a boolean type shared data, 
        # exposes low-latency boolean writing and reading methods

        self.rhc_status = RhcStatus(is_server=True,
                            cluster_size=self.cluster_size,
                            namespace=self.namespace, 
                            verbose=self._verbose, 
                            vlevel=VLevel.V2,
                            force_reconnection=False)
        
        self.cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                    is_server=False, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2, 
                                    safe=True,
                                    force_reconnection=False)
        
        # giving to handshaker all things which need to run in background
        self._handshaker = HandShaker([self.cluster_stats])

    def _trigger_solution(self):
        
        # # triggers all controllers
        # self.rhc_status.trigger.fill_with(True)
        # # writes whole internal view to shared memory
        # self.rhc_status.trigger.synch_all(read=False, 
        #                                 wait=True) # wait for synch to succeed
        
        # only trigger active controllers
        self.rhc_status.activation_state.synch_all(read=True, 
                                        wait=True)
        self.rhc_status.trigger.write_wait(self.rhc_status.activation_state.torch_view,
                                    0, 0)
        
    def _wait_for_solution(self):

        solved = False
        
        while not solved: 
            
            # fills views reading from shared memory
            self.rhc_status.activation_state.synch_all(read=True, 
                                        wait=True)
            self.rhc_status.trigger.synch_all(read=True, 
                                        wait=True) # wait for synch to succeed
            
            # only wait solution from active controllers
            solved_and_active = ~(self.rhc_status.trigger.torch_view[self.rhc_status.activation_state.torch_view])

            if (not self._terminate) and \
                (solved_and_active.shape[0] == 1 
                ):
                
                solved = (solved_and_active).all()

                self.perf_timer.clock_sleep(1000) # nanoseconds (but this
                # accuracy cannot be reached on a non-rt system)
                # on a modern laptop, this sleeps for about 5e-5s, but it does
                # so in a CPU-cheap manner

                continue

            else:

                # if no controller is active or the cluster is terminated we exit 

                break
        
        return solved
                
    def _compute_n_control_actions(self):

        if self.cluster_dt < self.control_dt:
            
            Journal.log(self.__class__.__name__,
                    "_compute_n_control_actions",
                    "cluster_dt has to be >= control_dt",
                    LogType.WARN,
                    throw_when_excep = False)

            self.n_sim_step_per_cntrl = 1
        
        else:
            
            self.n_sim_step_per_cntrl = round(self.cluster_dt / self.control_dt)
            self.cluster_dt = self.control_dt * self.n_sim_step_per_cntrl

        message = "The cluster controllers will run at a rate of " + \
                str(1.0 / self.cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self.control_dt) + "Hz.\n" + \
                "Number of sim steps per control step: " + str(self.n_sim_step_per_cntrl)

        Journal.log(self.__class__.__name__,
                "_compute_n_control_actions",
                message,
                LogType.INFO,
                throw_when_excep = False)
    
    def _finalize_init(self):
        
        Journal.log(self.__class__.__name__,
                "_compute_n_control_actions",
                "connecting to server...",
                LogType.INFO,
                throw_when_excep = False)
        
        # things to be done when everything is set but before starting to solve

        self.rhc_task_refs = RhcClusterTaskRefs(n_contacts=4, 
                                    cluster_size=self.cluster_size, 
                                    namespace=self.namespace,
                                    device=self._device, 
                                    backend=self._backend, 
                                    dtype=self.torch_dtype)
        self.rhc_task_refs.start()

        # also start cluster debug data client

        # self.cluster_stats.run()
        
        # self.cluster_stats.write_info(dyn_info_name=[
        #                                 "cluster_nominal_dt"],
        #                 val=[self.cluster_dt,
        #                     ])
        
        Journal.log(self.__class__.__name__,
                "_compute_n_control_actions",
                "connection achieved.",
                LogType.INFO,
                throw_when_excep = False)