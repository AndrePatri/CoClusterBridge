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
import torch

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
            vlevel: VLevel = VLevel.V1,
            debug = False, 
            force_reconnection: bool = False,
            use_pollingbased_waiting: bool = False):

        self.namespace = namespace
        
        self._verbose = verbose
        self._vlevel = vlevel

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
        if self.using_gpu:
            self._torch_device = torch.device("cuda")
        # shared mem objects

        self._robot_states = None
        self._rhc_cmds = None
        self._rhc_refs = None
        self._rhc_status = None
        self._cluster_stats = None 
        self._remote_triggerer = None
        self._remote_triggerer_ack_timeout = 60000 # [ns]

        # flags
        self._was_running = False
        self._is_running = False
        self._is_first_control_step = False
        self._pre_trigger_steps_done = False
        self._post_wait_steps_done = True
        
        self._triggered = False

        self.now_active = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device=self._torch_device)
        self.registered = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device=self._torch_device)
        self.prev_active_controllers = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device=self._torch_device)
        self.failed = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device=self._torch_device)

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
                                        retry=True)
        self._rhc_status.activation_state.synch_all(read=True, 
                                        retry=True)
        
        if self.using_gpu:
            self._rhc_status.registration.synch_mirror(from_gpu=False)
            self._rhc_status.activation_state.synch_mirror(from_gpu=False)
            
        # all active controllers will be triggered
        self.registered[:, :] = self._rhc_status.registration.get_torch_view(gpu=self.using_gpu)

        self.now_active[:, :] = self._rhc_status.activation_state.get_torch_view(gpu=self.using_gpu) & \
                            self._rhc_status.registration.get_torch_view(gpu=self.using_gpu) # controllers have to be registered
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

            self._rhc_status.controllers_counter.synch_all(retry = True,
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

    # def _trigger_solution(self):
        
    #     active_idxs = self.get_active_controllers()
    #     if active_idxs is not None:
    #         # at least one controller is active
    #         self._rhc_status.trigger.write_retry(self.now_active, # trigger active controllers
    #                                 0, 0)
    #         self._remote_triggerer.trigger() # signal to listening controllers
    
    def _trigger_solution(self):
        
        # trigger all
        trigger = self._rhc_status.trigger.get_torch_view()
        trigger[:, :] = True
        self._rhc_status.trigger.synch_all(read=False, retry=True)
        
        self._remote_triggerer.trigger() # signal to listening controllers

    def triggered(self):

        return self._triggered
    
    def wait_for_solution(self):

        if self.is_running():
            
            self._require_trigger() # we force sequentiality between triggering and
            # solution retrieval

            # we wait for controllers to finish   
            self._wait_for_solution()
            # at this point all controllers are done -> we synchronize the control commands on GPU
            # with the ones written by each controller on CPU

            self._get_rhc_sol() # not super efficient, but safe: in theory we should read only from 
            # controllers which where triggered 

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
    
    def _wait_for_solution(self):

        if not self._remote_triggerer.wait_ack_from(self.cluster_size, 
                                self._remote_triggerer_ack_timeout):
            Journal.log(self.__class__.__name__,
                "_wait_for_solution",
                f"Didn't receive any or all acks from controllers (expected {self.cluster_size})!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        # update flags (written by controllers upon solution request)
        self._rhc_status.fails.synch_all(read=True,
                                    retry=True)
        if self.using_gpu:
            self._rhc_status.fails.synch_mirror(from_gpu=False)
        self.failed[:,:] = self._rhc_status.fails.get_torch_view(gpu=self.using_gpu)

    def reset_controllers(self,
                    idxs: torch.Tensor = None):
        
        resets = self._rhc_status.resets.get_numpy_view()

        if idxs is not None:
            # write a reset request
            resets[idxs, :] = np.full(size=(idxs.shape[0], 1), 
                                    dtype=np.bool_,
                                    fill_value=True)
            self._rhc_status.resets.synch_all(read=False, retry=True)
        else:
            # resets all controllers
            resets[:, :] = True
            self._rhc_status.resets.synch_all(read=False, retry=True)
        
        self._remote_triggerer.trigger() # signal to listening controllers
        if not self._remote_triggerer.wait_ack_from(self.cluster_size, 
                                    self._remote_triggerer_ack_timeout):
            Journal.log(self.__class__.__name__,
                "reset_controllers",
                f"Didn't receive any or all acks from controllers (expected {self.cluster_size})!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        self._get_rhc_sol() # not super efficient, but safe: in theory we should only update the 
        # sol of the controllers which where reset 

    def activate_controllers(self,
                    idxs: torch.Tensor = None):
        
        if idxs is not None:
            activations = self._rhc_status.activation_state.get_numpy_view()
            activations[idxs, :] = np.full(size=(idxs.shape[0], 1), 
                                    fill_value=True,
                                    dtype=np.bool_)
            self._rhc_status.activation_state.synch_all(read=False, retry=True)
      
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

        n_controller = self._rhc_status.controllers_counter.get_numpy_view()
        self._n_controllers_connected = n_controller[0, 0].item()

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
        
        active = self._rhc_status.activation_state.get_numpy_view()
        if not active.all():
            msg = f"Controllers waiting to be activated... (" + \
                f"{self._rhc_status.activation_state.get_torch_view().sum().item()}/{self.cluster_size} active)"
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
                                with_torch_view=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=self._vlevel,
                                safe=False)

        self._rhc_cmds = RhcCmds(namespace=self.namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self.n_contact_sensors,
                                jnt_names=self.jnt_names,
                                contact_names=self.contact_linknames,
                                with_gpu_mirror=True,
                                with_torch_view=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=self._vlevel,
                                safe=False)

        self._rhc_refs = RhcRefs(namespace=self.namespace,
                            is_server=True,
                            n_robots=self.cluster_size,
                            n_jnts=self.n_dofs,
                            n_contacts=self.n_contact_sensors,
                            jnt_names=self.jnt_names,
                            contact_names=self.contact_linknames,
                            with_gpu_mirror = False,
                            with_torch_view=False,
                            force_reconnection = self._force_reconnection,
                            safe = False,
                            verbose = True,
                            vlevel = self._vlevel,
                            fill_value=np.nan)
        
        self._rhc_status = RhcStatus(is_server=True,
                            cluster_size=self.cluster_size,
                            namespace=self.namespace, 
                            verbose=self._verbose, 
                            vlevel=self._vlevel,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=False,
                            with_torch_view=True)
        
        cluster_info_dict = {}
        cluster_info_dict["cluster_size"] = self.cluster_size
        cluster_info_dict["cluster_dt"] = self.cluster_dt
        cluster_info_dict["low_level_control_dt"] = self.control_dt
        
        self._cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                    param_dict=cluster_info_dict,
                                    is_server=True, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=self._vlevel, 
                                    safe=True,
                                    force_reconnection=self._force_reconnection)

        self._remote_triggerer = RemoteTriggererSrvr(namespace=self.namespace,
                                            verbose=self._verbose,
                                            vlevel=self._vlevel,
                                            force_reconnection=self._force_reconnection)
        self._remote_triggerer.run()
                
        self._robot_states.run()

        self._rhc_cmds.run()

        self._rhc_refs.run()

        self._rhc_status.run()
        
        self._cluster_stats.run()          
    
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