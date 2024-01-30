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

from control_cluster_bridge.utilities.control_cluster_defs import RobotClusterState, RobotClusterCmd, RobotClusterContactState
from control_cluster_bridge.utilities.control_cluster_defs import HanshakeDataCntrlClient
from control_cluster_bridge.utilities.control_cluster_defs import RhcClusterTaskRefs

from control_cluster_bridge.utilities.data import RobotState

from control_cluster_bridge.utilities.shared_mem import SharedMemSrvr
from control_cluster_bridge.utilities.defs import launch_controllers_flagname

# from control_cluster_bridge.utilities.shared_cluster_info import SharedClusterInfo

from control_cluster_bridge.utilities.data import RHCStatus
from control_cluster_bridge.utilities.shared_info import ClusterStats

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

        # shared mem objects
        self.handshake_manager = None
        self._handshake_thread = None
        
        self.robot_states = None
        self.controllers_cmds = None

        self.launch_controllers = None
        self.rhc_task_refs = None
        self.contact_states = None
        # self.shared_cluster_info = None

        self.controller_status = None

        self.cluster_stats = None 
        
        # flags
        self._was_cluster_ready = False
        self.is_cluster_ready = False
        self._is_first_control_step = False
        self.controllers_active = False
        self.controllers_were_active = False
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
    
    def trigger_solution(self):

        # performs checks and triggers cluster solution

        handshake_done = self.handshake_manager.handshake_done

        self.controllers_active = self.launch_controllers.all()

        n_clients = self.controller_status.trigger.get_n_clients()

        if not handshake_done or \
            (n_clients == 0):
            
            if self._verbose and \
                    (self.trigger_counter+1) % self.n_steps_prints == 0: 
                
                Journal.log(self.__class__.__name__,
                    "trigger_solution",
                    "Waiting connection to ControlCluster server",
                    LogType.INFO,
                    throw_when_excep = False)

        if n_clients < self.cluster_size and \
                n_clients > 0 and \
                (self.trigger_counter+1) % self.n_steps_prints == 0:
            
            if self._verbose: 
                
                msg = f"Not all clients are connected yet ({n_clients}/{self.cluster_size})."

                Journal.log(self.__class__.__name__,
                    "trigger_solution",
                    msg,
                    LogType.WARN,
                    throw_when_excep = False)

        if self.controller_status.trigger.get_n_clients() > self.cluster_size:
            
            msg = f"More than {self.cluster_size} clients registered " + \
                "(total of {self.controller_status.trigger.get_n_clients()})." + \
                ". It's very likely a memory leak on the shared memory layer occurred." + \
                " You might need to reboot the system to clean the dangling memory."

            Journal.log(self.__class__.__name__,
                "trigger_solution",
                msg,
                LogType.EXCEP,
                throw_when_excep = True)
            
        if self._is_first_control_step:
                
            self._is_first_control_step = False
                
        if (not self._was_cluster_ready) and handshake_done:
            
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
            
            # updates shared tensor on CPU with latest data from states on GPU
            self.robot_states.synch_mirror(from_gpu=True) 

        if self.contact_states is not None:
            
            # updates shared tensor on CPU with latest contact data from the simulator
            # (possibly on GPU)
            self.contact_states.synch() 

        if self.is_cluster_ready:
            
            if self.controllers_active:
                
                self._trigger_solution() # actually triggers solution of all controllers in the cluster 
                # using latest state
            
            else:
            
                if self._verbose and \
                    (self.trigger_counter+1) % self.n_steps_prints == 0: 

                    Journal.log(self.__class__.__name__,
                    "trigger_solution",
                    "Controllers waiting to be activated...",
                    LogType.INFO,
                    throw_when_excep = False)

        self.trigger_counter +=1

    def _on_failure(self):

        # checks failure status
        self.controller_status.fails.synch_all(read=True, 
                                    wait=True)
        
        # writes failures to reset flags
        self.controller_status.resets.write_wait(self.controller_status.fails.torch_view,
                                    0, 0)
    
    def wait_for_solution(self):

        # will only return True after the solution signal from the cluster is received
        # or, if the cluster is inactive, will simply return False

        if self.is_cluster_ready:
            
            if self.controllers_active:

                # we wait for all controllers to finish      
                self._wait_for_solution() # this is blocking (but no busy wait)
                
                # at this point we are sure solutions and controllers status are updated

                # we handle controller fails
                self._on_failure()

                # at this point all controllers are done -> we synchronize the control commands on GPU
                # with the ones written by each controller on CPU
                self.controllers_cmds.synch()
            
            else:
            
                if self._verbose and \
                    (self.wait_for_sol_counter+1) % self.n_steps_prints == 0: 

                    Journal.log(self.__class__.__name__,
                    "wait_for_solution",
                    "Controllers waiting to be activated...",
                    LogType.INFO,
                    throw_when_excep = False)

            self.solution_counter += 1

        if self._debug and self.is_cluster_ready:
            
            # this is updated even if the cluster is not active (values do not make sense
            # in that case)

            self.solution_time = time.perf_counter() - self.start_time # we profile the whole solution pipeline
            
            # self.shared_cluster_info.update(solve_time=self.solution_time, 
            #                             controllers_up = self.controllers_active) # we update the shared info

            self.cluster_stats.write_info(dyn_info_name=["cluster_sol_time", 
                                        "cluster_rt_factor",
                                        "cluster_ready",
                                        "cluster_state_update_dt"],
                        val=[self.solution_time,
                            self.cluster_dt/self.solution_time,
                            self.cluster_ready(),
                            np.nan])
    
        self.controllers_were_active = self.controllers_active

        self.wait_for_sol_counter +=1

    def close(self):
        
        if not self._terminate:
            
            self._terminate = True
            
            if self.robot_states is not None:
                
                self.robot_states.close()

            if self.launch_controllers is not None:

                self.launch_controllers.terminate()

            if self.controllers_cmds is not None:
                
                self.controllers_cmds.terminate()

            if self.rhc_task_refs is not None:

                self.rhc_task_refs.terminate()

            if self.contact_states is not None:

                self.contact_states.terminate()

            if self.controller_status is not None:

                self.controller_status.close()
            
            if self.cluster_stats is not None:

                self.cluster_stats.close()

        self._close_handshake()

    def _close_handshake(self):

        if self.handshake_manager is not None:
                
                self.handshake_manager.terminate() # will close/detach all shared memory 
                
        if self._handshake_thread is not None:
                
            self._close_thread(self._handshake_thread) # we first wait for thread to exit, if still alive

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

        if self.contact_states is not None:

            self.contact_states.start()

        self.launch_controllers.start()
        self.launch_controllers.reset_bool(False)

        # self.shared_cluster_info.start()

        self.controller_status.run()

        self._spawn_handshake() # we launch all the child processes

    def _spawn_handshake(self):
        
        # we spawn the heartbeat() to another process, 
        # so that it's not blocking wrt the simulator

        self._handshake_thread =  threading.Thread(target=self.handshake_manager.start, 
                                args=(self.cluster_size, self.jnt_names, ), 
                                kwargs={})
        
        self._handshake_thread.start()

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
                                with_gpu_mirror=True,
                                force_reconnection=False,
                                verbose=True,
                                vlevel=VLevel.V2,
                                safe=False)

        # self.robot_states = RobotClusterState(n_dofs=self.n_dofs, 
        #                                     cluster_size=self.cluster_size, 
        #                                     namespace=self.namespace,
        #                                     backend=self._backend, 
        #                                     device=self._device, 
        #                                     dtype=self.torch_dtype) # from robot to controllers

        if not self.n_contact_sensors < 0:
            self.contact_states = RobotClusterContactState(cluster_size=self.cluster_size,
                                        n_contacts=self.n_contact_sensors, 
                                        contact_names=self.contact_linknames, 
                                        namespace=self.namespace,
                                        backend=self._backend, 
                                        device=self._device, 
                                        dtype=self.torch_dtype, 
                                        verbose=self._verbose)
                                            
        self.handshake_manager = HanshakeDataCntrlClient(n_jnts=self.n_dofs, 
                                                    namespace=self.namespace) # handles handshake process
        # between client and server

        dtype = torch.bool # using a boolean type shared data, 
        # exposes low-latency boolean writing and reading methods
        
        self.launch_controllers = SharedMemSrvr(n_rows=1, 
                                    n_cols=1, 
                                    name=launch_controllers_flagname(), 
                                    namespace=self.namespace,
                                    dtype=dtype) 

        self.controller_status = RHCStatus(is_server=True,
                                        cluster_size=self.cluster_size,
                                        namespace=self.namespace, 
                                        verbose=self._verbose, 
                                        vlevel=VLevel.V2,
                                        force_reconnection=False)
        
        
        self.cluster_stats = ClusterStats(cluster_size=self.cluster_size,
                                    is_server=False, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2, 
                                    safe=True,
                                    force_reconnection=False)

    def _trigger_solution(self):
        
        # fills view to trigger controllers
        self.controller_status.trigger.fill_with(True)

        # writes whole internal view to shared memory
        self.controller_status.trigger.synch_all(read=False, 
                                        wait=True) # wait for synch to succeed
            
    def _wait_for_solution(self):

        solved = False
        
        while not solved: 
            
            # fills views reading from shared memory
            self.controller_status.trigger.synch_all(read=True, 
                                        wait=True) # wait for synch to succeed
            
            solved = (~self.controller_status.trigger.torch_view).all()

            if (not self._terminate) and \
                (self.controller_status.trigger.get_n_clients() == self.cluster_size):
                
                self.perf_timer.clock_sleep(1000) # nanoseconds (but this
                # accuracy cannot be reached on a non-rt system)
                # on a modern laptop, this sleeps for about 5e-5s, but it does
                # so in a CPU-cheap manner

                continue
            
            else:
                
                solved = False

                break
        
        return
                
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

        add_data_length_from_server = self.handshake_manager.add_data_length.tensor_view[0, 0].item()
        n_contacts_from_server = self.handshake_manager.n_contacts.tensor_view[0, 0].item()

        self.controllers_cmds = RobotClusterCmd(n_dofs=self.n_dofs, 
                                            cluster_size=self.cluster_size,
                                            namespace=self.namespace,
                                            add_data_size = add_data_length_from_server, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # now that we know add_data_size
        # we can initialize the control commands
        self.controllers_cmds.start()

        self.rhc_task_refs = RhcClusterTaskRefs(n_contacts=n_contacts_from_server, 
                                    cluster_size=self.cluster_size, 
                                    namespace=self.namespace,
                                    device=self._device, 
                                    backend=self._backend, 
                                    dtype=self.torch_dtype)
        self.rhc_task_refs.start()

        # also start cluster debug data client

        self.cluster_stats.run()
        
        self.cluster_stats.write_info(dyn_info_name=[
                                        "cluster_nominal_dt"],
                        val=[self.cluster_dt,
                            ])
        
        Journal.log(self.__class__.__name__,
                "_compute_n_control_actions",
                "connection achieved.",
                LogType.INFO,
                throw_when_excep = False)