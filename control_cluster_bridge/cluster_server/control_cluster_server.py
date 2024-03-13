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

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState 
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling
from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererSrvr

from SharsorIPCpp.PySharsorIPC import VLevel, Journal, LogType

import time

import numpy as np

from typing import List

from perf_sleep.pyperfsleep import PerfSleep

class ControlClusterServer(ABC):

    def __init__(self, 
            namespace: str,
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            jnt_names: List[str],
            n_contact_sensors: int = -1,
            contact_linknames: List[str] = None,
            use_gpu: bool = False, 
            verbose = False, 
            debug = False, 
            force_reconnection: bool = False,
            use_pollingbased_waiting: bool = False):

        self.namespace = namespace
        
        self._verbose = verbose

        self._closed = False

        self._debug = debug
        
        self._use_pollingbased_waiting = use_pollingbased_waiting

        self.jnt_names = jnt_names

        self.n_dofs = len(self.jnt_names)
        
        self.cluster_size = cluster_size
        
        self._n_controllers_connected = 0

        self.jnt_names = jnt_names

        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs
     
        self.using_gpu = use_gpu

        # shared mem objects

        self._robot_states = None
        self._rhc_cmds = None
        self._rhc_refs = None
        self._rhc_status = None
        self._cluster_stats = None 
        self._remote_triggerer = None
        self._remote_triggerer_ack_timeout = 10000 # [ns]

        # flags
        self._was_running = False
        self._is_running = False
        self._is_first_control_step = False
        self._pre_trigger_steps_done = False
        self._post_wait_steps_done = True
        
        self._triggered = False

        self.now_active = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool)
        self.registered = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool)
        self.prev_active_controllers = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool)
        self.failed = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool)

        # other data
        self.n_contact_sensors = n_contact_sensors
        self.contact_linknames = contact_linknames

        self.solution_time = np.nan
        self.start_time = np.nan
        self.solution_counter = 0
        self.n_sim_step_per_cntrl = -1

        self.trigger_counter = 0 # used for debug purposes
        self.wait_for_sol_counter = 0
        self._print_frequency = 100 # number of "steps" at which sporadic logs are printed

        self._force_reconnection = force_reconnection

    def __del__(self):
                
        self.close()

    def run(self):

        self._setup()

        self._is_running = True

    def close(self):
        
        if not self._closed:
            
            self._closed = True
            
            if self._robot_states is not None:
                
                self._robot_states.close()

            if self._rhc_cmds is not None:
                
                self._rhc_cmds.close()
            
            if self._rhc_refs is not None:
                
                self._rhc_refs.close()

            if self._rhc_status is not None:

                self._rhc_status.close()
            
            if self._cluster_stats is not None:

                self._cluster_stats.close()

            if self._remote_triggerer is not None:
                
                self._remote_triggerer.close()

    def n_controllers(self):

        return self._n_controllers_connected
    
    def pre_trigger_steps(self):

        # first retrive current controllers status
        self._rhc_status.registration.synch_all(read=True,
                                        wait=True)
        
        self._rhc_status.activation_state.synch_all(read=True, 
                                        wait=True)
                
        # all active controllers will be triggered
        self.registered[:, :] = self._rhc_status.registration.torch_view

        self.now_active[:, :] = self._rhc_status.activation_state.torch_view & \
                            self._rhc_status.registration.torch_view # controllers have to be registered
                            # to be considered active
        
        
        self._pre_trigger_steps_done = True

    def pretriggered(self):

        return self._pre_trigger_steps_done
    
    def trigger_solution(self):

        # performs checks and triggers cluster solution

        if self.is_running():

            if self._debug:
                
                # we profile the whole solution pipeline
                self.start_time = time.perf_counter() 

            self._require_pretrigger() # we force sequentiality between pretriggering and
            # solution triggering

            self._rhc_status.controllers_counter.synch_all(wait = True,
                                                        read = True)

            self._pre_trigger_logs() # debug info
                
            if self._is_first_control_step:
                    
                self._is_first_control_step = False
                    
            if (not self._was_running):
                
                # first time the cluster is ready 

                self._is_first_control_step = True

            if self.using_gpu:

                # updates shared tensor on CPU with latest data from states on GPU
                # and writes to shared mem
                self._robot_states.synch_mirror(from_gpu=True)
            
            else:
                
                # only write to shared mem (gpu mirror is not used)
                self._robot_states.synch_to_shared_mem()
                                
            self._trigger_solution() # actually triggers solution of all controllers in the cluster 
            # which are ACTIVE using the latest available state
            
            self._post_trigger_logs() # debug info

            self.trigger_counter +=1

            self._triggered = True

    def triggered(self):

        return self._triggered
    
    def wait_for_solution(self):

        if self.is_running():
            
            self._require_trigger() # we force sequentiality between triggering and
            # solution retrieval

            done = False
            # we wait for controllers to finish   
            if not self._use_pollingbased_waiting:
                done = self._wait_for_solution()
            else:
                done = self._wait_for_solution_withpolling() # this is blocking if at least a controller
            # is active

            # at this point all controllers are done -> we synchronize the control commands on GPU
            # with the ones written by each controller on CPU

            if done:
                self._get_rhc_sol()
                self.solution_counter += 1
            
            self._triggered = False # allow next trigger

            if self._debug:
                self.solution_time = time.perf_counter() - self.start_time # we profile the whole solution pipeline
                # update shared debug info
                self._cluster_stats.write_info(dyn_info_name=["cluster_sol_time", 
                                            "cluster_rt_factor",
                                            "cluster_ready"],
                            val=[self.solution_time,
                                self.cluster_dt/self.solution_time,
                                self.is_running()])

        self._update_mem_flags() # used to keep track of previous flag states
    
    def get_actions(self):

        return self._rhc_cmds
    
    def get_state(self):

        return self._robot_states
    
    def get_refs(self):

        return self._rhc_refs
    
    def get_status(self):

        return self._rhc_status
    
    def get_stats(self):

        return self._cluster_stats
        
    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index is the current simulation loop number (0-based)
        
        # returns true if this is a control "instant"
        
        return (control_index + 1) % self.n_sim_step_per_cntrl == 0
    
    def get_just_activated(self):
        
        # gets indexes of controllers which are triggered for the first time
        # after being activated
        now_active = self.now_active.squeeze(dim=1)
        not_active_before = ~self.prev_active_controllers.squeeze(dim=1)

        just_activated = torch.nonzero(now_active & not_active_before).squeeze(dim=1)
        
        if not just_activated.shape[0] == 0:
            
            return just_activated
        
        else:
            
            # no controller just activated

            return None
        
    def get_just_deactivated(self):
        
        # gets indexes of controllers which are triggered for the first time
        # after being activated
        now_not_active = ~self.now_active.squeeze(dim=1)
        active_before = self.prev_active_controllers.squeeze(dim=1)

        just_deactivated = torch.nonzero(now_not_active & active_before).squeeze(dim=1)
    
        if not just_deactivated.shape[0] == 0:
            
            return just_deactivated
        
        else:
            
            # no controller just deactivated

            return None
    
    def get_active_controllers(self):
        
        now_active = torch.nonzero(self.now_active.squeeze(dim=1)).squeeze(dim=1)
        
        if not now_active.shape[0] == 0:
  
            return now_active
        
        else:
            
            # no controller active

            return None
    
    def get_inactive_controllers(self):
        
        not_active = torch.nonzero((~self.now_active).squeeze(dim=1)).squeeze(dim=1)
        
        if not not_active.shape[0] == 0:
  
            return not_active
        
        else:
            
            # no controller active

            return None

    def get_failed_controllers(self):
        
        failed = torch.nonzero(self.failed.squeeze(dim=1)).squeeze(dim=1)
        
        if not failed.shape[0] == 0:
  
            return failed
        
        else:
            
            # no controller active

            return None
    
    def get_registered_controllers(self):

        registered = torch.nonzero(self.registered.squeeze(dim=1)).squeeze(dim=1)
        
        if not registered.shape[0] == 0:
  
            return registered
        
        else:
            
            # no controller registered

            return None
        
    def just_started_running(self):

        return self.is_running() and (not self._was_running)
    
    def reset_controllers(self,
                    idxs: torch.Tensor = None):
        
        if idxs is not None:
            
            # untrigger, just in case
            self._rhc_status.trigger.torch_view[idxs, :] = torch.full(size=(idxs.shape[0], 1), 
                                                            fill_value=False) 
            self._rhc_status.trigger.synch_all(read=False, wait=True)

            # reset
            self._rhc_status.resets.torch_view[idxs, :] = torch.full(size=(idxs.shape[0], 1), 
                                                            fill_value=True)
            
            self._rhc_status.resets.synch_all(read=False, wait=True)

        else:
            
            # untrigger all controllers
            self._rhc_status.trigger.torch_view[idxs, :] = torch.full(size=(self.cluster_size, 1), 
                                                            fill_value=False) 
            self._rhc_status.trigger.synch_all(read=False, wait=True)

            # resets all failed controllers
            self._rhc_status.resets.write_wait(self._rhc_status.fails.torch_view,
                                        0, 0)
        
        self._wait_for_reset_done() # wait for any pending reset request to be completed

        self._get_rhc_sol() # not efficient: in theory we should only update the 
        # sol of the reset controllers

    def activate_controllers(self,
                    idxs: torch.Tensor = None):
        
        if idxs is not None:

            self._rhc_status.activation_state.torch_view[idxs, :] = torch.full(size=(idxs.shape[0], 1), 
                                                            fill_value=True)
            
            self._rhc_status.activation_state.synch_all(read=False, wait=True)
            
    def is_running(self):

        return self._is_running
    
    def is_first_control_step(self):

        return self._is_first_control_step
    
    def _get_rhc_sol(self):

        if self.using_gpu:
                    
            # first reads cmds from shared memory and then synchs gpu mirror with the read data
            self._rhc_cmds.synch_mirror(from_gpu=False) 

        else:
            
            # only read cmds from shared mem
            self._rhc_cmds.synch_from_shared_mem()

    def _require_pretrigger(self):

        if not self.pretriggered():

            exception = "Cannot call trigger() if pre_trigger_steps()" + \
                " was not previously called!"

            Journal.log(self.__class__.__name__,
                "_require_pretrigger",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def _require_trigger(self):

        if not self.triggered():

            exception = "Cannot call wait_for_solution() if trigger_solution()" + \
                " was not previously called!"

            Journal.log(self.__class__.__name__,
                "_require_trigger",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

    def _update_mem_flags(self):

        self.prev_active_controllers[:, :] = self.now_active

        self._was_running = self._is_running

        self.wait_for_sol_counter +=1

    def _pre_trigger_logs(self):

        self._n_controllers_connected = self._rhc_status.controllers_counter.torch_view[0, 0].item()

        if self._n_controllers_connected == 0:
            
            self._sporadic_log(calling_methd="trigger_solution",
                        msg = "waiting connection to ControlCluster server")

        if self._n_controllers_connected < self.cluster_size and \
                self._n_controllers_connected > 0 and \
                (self.trigger_counter+1) % self._print_frequency == 0:
                                            
            self._sporadic_log(calling_methd="trigger_solution",
                    msg = f"Not all clients are connected yet ({self._n_controllers_connected}/{self.cluster_size}).",
                    logtype=LogType.WARN)

        if self._n_controllers_connected > self.cluster_size:
            
            msg = f"More than {self.cluster_size} controllers registered " + \
                f"(total of {self._n_controllers_connected})." + \
                ". It's very likely a memory leak on the shared memory layer occurred." + \
                " You might need to reboot the system to clean the dangling memory."

            self._sporadic_log(calling_methd="trigger_solution",
                        msg = msg,
                        logtype=LogType.EXCEP)

    def _post_trigger_logs(self):

        if not self._rhc_status.activation_state.torch_view.all():
                
                msg = f"Controllers waiting to be activated... (" + \
                    f"{self._rhc_status.activation_state.torch_view.sum().item()}/{self.cluster_size} active)"
                
                self._sporadic_log(calling_methd="trigger_solution",
                            msg = msg,
                            logtype=LogType.INFO)

    def _sporadic_log(self,
                calling_methd: str,
                msg: str,
                logtype: LogType = LogType.INFO):

        if self._verbose and \
            (self.trigger_counter+1) % self._print_frequency == 0: 
            
            Journal.log(self.__class__.__name__,
                calling_methd,
                msg,
                logtype,
                throw_when_excep = True)
                
    def _setup(self):

        self._compute_n_control_actions() # necessary ti apply control input only at 
        # a specific rate

        self._setup_shared_mem() # initializes shared memory used for 
        # communication between the client and server

    def _setup_shared_mem(self):

        self._robot_states = RobotState(namespace=self.namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self.n_contact_sensors,
                                jnt_names=self.jnt_names,
                                contact_names=self.contact_linknames,
                                with_gpu_mirror=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=VLevel.V2,
                                safe=False)

        self._rhc_cmds = RhcCmds(namespace=self.namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self.n_contact_sensors,
                                jnt_names=self.jnt_names,
                                contact_names=self.contact_linknames,
                                with_gpu_mirror=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=VLevel.V2,
                                safe=False)

        self._rhc_refs = RhcRefs(namespace=self.namespace,
                            is_server=True,
                            n_robots=self.cluster_size,
                            n_jnts=self.n_dofs,
                            n_contacts=self.n_contact_sensors,
                            jnt_names=self.jnt_names,
                            contact_names=self.contact_linknames,
                            with_gpu_mirror = True,
                            force_reconnection = self._force_reconnection,
                            safe = False,
                            verbose = True,
                            vlevel = VLevel.V2,
                            fill_value=np.nan)
        
        self._rhc_status = RhcStatus(is_server=True,
                            cluster_size=self.cluster_size,
                            namespace=self.namespace, 
                            verbose=self._verbose, 
                            vlevel=VLevel.V2,
                            force_reconnection=self._force_reconnection)
        
        cluster_info_dict = {}
        cluster_info_dict["cluster_size"] = self.cluster_size
        cluster_info_dict["cluster_dt"] = self.cluster_dt
        cluster_info_dict["low_level_control_dt"] = self.control_dt
        
        self._cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                    param_dict=cluster_info_dict,
                                    is_server=True, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2, 
                                    safe=True,
                                    force_reconnection=self._force_reconnection)

        self._remote_triggerer = RemoteTriggererSrvr(namespace=self.namespace,
                                            verbose=self._verbose,
                                            vlevel=VLevel.V2,
                                            force_reconnection=self._force_reconnection)
        self._remote_triggerer.run()
        
        self._robot_states.run()

        self._rhc_cmds.run()

        self._rhc_refs.run()

        self._rhc_status.run()
        
        self._cluster_stats.run()          

    def _trigger_solution(self):

        self._rhc_status.trigger.write_wait(self.now_active, # trigger active controllers
                                    0, 0)
        if not self._use_pollingbased_waiting:
            self._remote_triggerer.trigger() # trigger all listening consumers
        
    def _wait_for_solution_withpolling(self):

        solved = False
        
        while not solved: 
            
            # fills views reading from shared memory
            self._rhc_status.activation_state.synch_all(read=True, 
                                        wait=True)
            self._rhc_status.trigger.synch_all(read=True, 
                                        wait=True) # wait for synch to succeed
            self._rhc_status.fails.synch_all(read=True,
                                        wait=True)
            
            self.failed[:,:] = self._rhc_status.fails.torch_view 

            active_and_not_failed_controllers = torch.logical_and(self._rhc_status.activation_state.torch_view,
                                                        ~self._rhc_status.fails.torch_view)

            # only wait solution from active (and not failed) controllers
            solved_and_ok = ~(self._rhc_status.trigger.torch_view[active_and_not_failed_controllers])

            active_idxs = torch.nonzero(self.now_active.squeeze(dim=1)).squeeze(dim=1)

            if (not self._closed) and \
                (solved_and_ok.shape[0] == active_idxs.shape[0] # all active controllers have solved
                ):
                
                # wait for all triggered (i.e. active) controllers to finish
                solved = (solved_and_ok).all()

                PerfSleep.thread_sleep(1000) # nanoseconds (but this
                # accuracy cannot be reached on a non-rt system)
                # on a modern laptop, this sleeps for about 5e-5s, but it does
                # so in a CPU-cheap manner

                continue

            else:

                # if no controller is active or the cluster is terminated we exit 

                break
        
        return solved

    def _wait_for_solution(self):

        # active controllers were triggered -> we
        # wait for a response from them
        active_idxs = self.get_active_controllers()
        if active_idxs is not None:
            if not self._remote_triggerer.wait_ack_from(active_idxs.shape[0], 
                                    self._remote_triggerer_ack_timeout):
                Journal.log(self.__class__.__name__,
                    "_wait_for_solution",
                    f"Didn't receive any or all acks from controllers (expected {active_idxs.shape[0]})!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self._rhc_status.activation_state.synch_all(read=True, 
                                        wait=True)
            self._rhc_status.trigger.synch_all(read=True, 
                                        wait=True) # wait for synch to succeed
            self._rhc_status.fails.synch_all(read=True,
                                        wait=True)
            
            self.failed[:,:] = self._rhc_status.fails.torch_view 

            active_and_not_failed_controllers = torch.logical_and(self._rhc_status.activation_state.torch_view,
                                                        ~self._rhc_status.fails.torch_view)
            
            solved_and_ok = ~(self._rhc_status.trigger.torch_view[active_and_not_failed_controllers])

            return (solved_and_ok).all()
        
        else:

            return False
    
    def _wait_for_reset_done(self):

        # waits for requested controller resets to be completed

        done = False
        
        while not done: 
            
            # fills views reading from shared memory
            self._rhc_status.resets.synch_all(read=True, 
                                        wait=True)
            
            resets = self._rhc_status.resets.get_torch_view(gpu=False)

            if resets.any().item(): # controllers reset their 
                # correspoding reset flags to false when the reset is done
                
                PerfSleep.thread_sleep(1000) # nanoseconds

                continue

            else:

                done = True
            
        return done
    
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