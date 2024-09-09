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

class ControlClusterServer(ABC):

    def __init__(self, 
            namespace: str,
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            jnt_names: List[str],
            n_contacts: int,
            contact_linknames: List[str] = None,
            use_gpu: bool = False, 
            verbose = False, 
            vlevel: VLevel = VLevel.V1,
            debug = False, 
            force_reconnection: bool = False,
            timeout_ms: int = 60000):
        
        self._verbose = verbose
        self._vlevel = vlevel

        self._closed = False
        self._debug = debug

        self.jnt_names = jnt_names
        self.n_dofs = len(self.jnt_names)
        self.cluster_size = cluster_size

        self._cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self._low_level_control_dt = control_dt # dt at which the low level controller or the simulator runs
     
        self._using_gpu = use_gpu
        if self._using_gpu:
            self._torch_device = torch.device("cuda")

        # shared mem objects
        self._namespace = namespace # unique ID for shared memory and cluster
        self._force_reconnection = force_reconnection
        self._robot_states = None
        self._rhc_cmds = None
        self._rhc_refs = None
        self._rhc_status = None
        self._cluster_stats = None 
        self._remote_triggerer = None
        self._remote_triggerer_ack_timeout = timeout_ms # [ns]
        self._n_controllers_connected = 0

        # flags
        self._was_running = False
        self._is_running = False
        
        # no need for this flags to be on GPU (if necessary copies are made on demand)
        self._now_active = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device="cpu")
        self._registered = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device="cpu")
        self._prev_active_controllers = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device="cpu")
        self._failed = torch.full(fill_value=False, size=(self.cluster_size, 1), dtype=torch.bool, device="cpu")

        # other data
        self._n_contacts = n_contacts
        self._contact_linknames = contact_linknames
        if self._contact_linknames is None:
            self._contact_linknames=[]
            for i in range(self._n_contacts):
                self._contact_linknames.append(f"contact_{i+1}")
        self._solution_time = np.nan
        self._start_time = np.nan
        
        self._n_steps_per_cntrl = -1

        self._solution_counter = 0
        self._pre_trigger_counter = 0
        self._trigger_counter = 0

        self._print_frequency = 100 # number of "steps" at which sporadic logs are printed

    def __del__(self):     
        self.close()

    def run(self):
        self._setup()
        self._is_running = True

    def _setup(self):
        self._compute_n_control_actions() # necessary ti apply control input only at 
        # a specific rate
        self._setup_shared_mem() # initializes shared memory used for 
        # communication between the client and server

    def cluster_dt(self):
        return self._cluster_dt
    
    def _compute_n_control_actions(self):
        if self._cluster_dt < self._low_level_control_dt:
            Journal.log(self.__class__.__name__,
                    "_compute_n_control_actions",
                    "cluster_dt has to be >= control_dt",
                    LogType.EXCEP,
                    throw_when_excep = True)        
        else:
            self._n_steps_per_cntrl = round(self._cluster_dt / self._low_level_control_dt)
            self._cluster_dt = self._low_level_control_dt * self._n_steps_per_cntrl
        db_info = "The cluster controllers will run at a rate of " + \
                str(1.0 / self._cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self._low_level_control_dt) + "Hz.\n" + \
                "Number of sim steps per control step: " + str(self._n_steps_per_cntrl)
        Journal.log(self.__class__.__name__,
                "_compute_n_control_actions",
                db_info,
                LogType.INFO,
                throw_when_excep = False)
        
    def _setup_shared_mem(self):

        self._robot_states = RobotState(namespace=self._namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self._n_contacts,
                                jnt_names=self.jnt_names,
                                contact_names=self._contact_linknames,
                                with_gpu_mirror=True,
                                with_torch_view=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=self._vlevel,
                                safe=False)
        self._rhc_cmds = RhcCmds(namespace=self._namespace,
                                is_server=True,
                                n_robots=self.cluster_size,
                                n_jnts=self.n_dofs,
                                n_contacts=self._n_contacts,
                                jnt_names=self.jnt_names,
                                contact_names=self._contact_linknames,
                                with_gpu_mirror=True,
                                with_torch_view=True,
                                force_reconnection=self._force_reconnection,
                                verbose=True,
                                vlevel=self._vlevel,
                                safe=False)
        self._rhc_refs = RhcRefs(namespace=self._namespace,
                            is_server=True,
                            n_robots=self.cluster_size,
                            n_jnts=self.n_dofs,
                            n_contacts=self._n_contacts,
                            jnt_names=self.jnt_names,
                            contact_names=self._contact_linknames,
                            with_gpu_mirror = False,
                            with_torch_view=False,
                            force_reconnection = self._force_reconnection,
                            safe = False,
                            verbose = True,
                            vlevel = self._vlevel,
                            fill_value=np.nan)
        self._rhc_status = RhcStatus(is_server=True,
                            cluster_size=self.cluster_size,
                            n_nodes=100, # just an ub which should fit for most cases
                            n_contacts=self._n_contacts,
                            namespace=self._namespace, 
                            verbose=self._verbose, 
                            vlevel=self._vlevel,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=False,
                            with_torch_view=True)
        cluster_info_dict = {}
        cluster_info_dict["cluster_size"] = self.cluster_size
        cluster_info_dict["cluster_dt"] = self._cluster_dt
        cluster_info_dict["low_level_control_dt"] = self._low_level_control_dt
        self._cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                    param_dict=cluster_info_dict,
                                    is_server=True, 
                                    name=self._namespace,
                                    verbose=self._verbose,
                                    vlevel=self._vlevel, 
                                    safe=True,
                                    force_reconnection=self._force_reconnection)
        self._remote_triggerer = RemoteTriggererSrvr(namespace=self._namespace,
                                            verbose=self._verbose,
                                            vlevel=self._vlevel,
                                            force_reconnection=self._force_reconnection)
        self._remote_triggerer.run()
        self._robot_states.run()
        self._rhc_cmds.run()
        self._rhc_refs.run()
        self._rhc_status.run()
        self._cluster_stats.run()          
    
    def close(self):
        # close all shared memory
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
    
    def n_contact_sensors(self):
        return self._n_contacts
    
    def contact_linknames(self):
        return self._contact_linknames
    
    def solution_time(self):
        return self._solution_time
    
    def pretriggered(self):
        return (self._pre_trigger_counter - self._trigger_counter) == 1
    
    def triggered(self):
        return (self._trigger_counter - self._solution_counter) == 1
    
    def solution_counter(self):
        return self._solution_counter
    
    def trigger_counter(self):
        return self._trigger_counter
    
    def pre_trigger(self):
        # first retrieve current controllers status (this is a 
        # separate method wrt trigger_solution to allow higher level code
        # to perform operations in between depending on the controllers status) 
        if self._debug:
            self._check_running()
        self._rhc_status.registration.synch_all(read=True,
                                        retry=True)
        self._rhc_status.activation_state.synch_all(read=True, 
                                        retry=True)
        # all active controllers will be triggered
        self._registered[:, :] = self._rhc_status.registration.get_torch_mirror(gpu=False)
        self._prev_active_controllers[:, :] = self._now_active
        self._now_active[:, :] = self._rhc_status.activation_state.get_torch_mirror(gpu=False) & \
                            self._rhc_status.registration.get_torch_mirror(gpu=False) # controllers have to be registered
                            # to be considered active
        self._pre_trigger_counter +=1
    
    def trigger_solution(self):
        # performs checks and triggers cluster solution
        if self._debug:
            # we profile the whole solution pipeline
            self._check_running()
            self._start_time = time.perf_counter() 
            self._pre_trigger_logs() # debug info + checks
            self._require_pretrigger() # we force sequentiality between pretriggering and
            # solution triggering
        self._set_rhc_state() # set the state employed by the controllers in the cluster       
        self._trigger_solution() # triggers solution of all controllers in the cluster 
        # which are ACTIVE using the latest available state
        if self._debug:
            self._post_trigger_logs() # debug info
        self._trigger_counter +=1
    
    def _trigger_solution(self):
        # trigger all
        trigger = self._rhc_status.trigger.get_torch_mirror()
        trigger[:, :] = True
        self._rhc_status.trigger.synch_all(read=False, retry=True)
        self._remote_triggerer.trigger() # signal to listening controllers to process
        # request

    def wait_for_solution(self):
        if self._debug:
            self._check_running()
            self._require_trigger() # we force sequentiality between triggering and
            # solution retrieval
        self._wait_for_solution() # we wait for controllers to finish processing the trigger request
        self._get_rhc_sol() # not super efficient, but safe: in theory we should read solution only from 
        # controllers which where triggered (i.e. ACTIVE ones)
        if self._debug:
            self._solution_time = time.perf_counter() - self._start_time # we profile the whole solution pipeline
            # and update some shared debug info
            self._cluster_stats.write_info(dyn_info_name=["cluster_sol_time", 
                                        "cluster_rt_factor",
                                        "cluster_ready"],
                        val=[self._solution_time,
                            self._cluster_dt/self._solution_time,
                            self.is_running()])

        self._was_running = self._is_running
        self._solution_counter += 1
    
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
        self._failed[:,:] = self._rhc_status.fails.get_torch_mirror(gpu=False)

    def reset_controllers(self,
                    idxs: torch.Tensor = None):
        
        # set reset request
        resets = self._rhc_status.resets.get_torch_mirror()
        if idxs is not None:
            # write a reset request
            resets[idxs, :] = True
            self._rhc_status.resets.synch_all(read=False, retry=True)
        else:
            # resets all controllers
            resets[:, :] = True
            self._rhc_status.resets.synch_all(read=False, retry=True)
        # send signal to listening controllers to process request
        self._remote_triggerer.trigger() 
        if not self._remote_triggerer.wait_ack_from(self.cluster_size, 
                            self._remote_triggerer_ack_timeout):
            Journal.log(self.__class__.__name__,
                "reset_controllers",
                f"Didn't receive any or all acks from controllers (expected {self.cluster_size})!",
                LogType.EXCEP,
                throw_when_excep = True)
        
        self._get_rhc_sol() # not super efficient, but safe: in theory we should only update the 
        # sol of the controllers which where reset 

        self._rhc_status.resets.synch_all(read=True, retry=True) # update reset flags (controllers
        # reset flags upon successful reset)

    def activate_controllers(self,
                    idxs: torch.Tensor = None):
        if idxs is not None:
            activations = self._rhc_status.activation_state.get_torch_mirror()
            activations[idxs, :] = True
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
        # control_index is the current simulation loop iteration (0-based)
        # returns true if this is a control "instant"
        return (control_index + 1) % self._n_steps_per_cntrl == 0
    
    def get_just_activated(self,
                    gpu=False):
        
        # gets indexes of controllers which are triggered for the first time
        # after being activated
        now_active = self._now_active.squeeze(dim=1)
        not_active_before = ~self._prev_active_controllers.squeeze(dim=1)
        just_activated = torch.nonzero(now_active & not_active_before).squeeze(dim=1)
        
        if not just_activated.shape[0] == 0:
            if gpu:
                return just_activated.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return just_activated
        else:
            # no controller just activated
            return None
        
    def get_just_deactivated(self,
                    gpu=False):
        
        # gets indexes of controllers which are triggered for the first time
        # after being activated
        now_not_active = ~self._now_active.squeeze(dim=1)
        active_before = self._prev_active_controllers.squeeze(dim=1)
        just_deactivated = torch.nonzero(now_not_active & active_before).squeeze(dim=1)
        
        if not just_deactivated.shape[0] == 0:
            if gpu:
                return just_deactivated.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return just_deactivated        
        else:
            # no controller just deactivated
            return None
    
    def get_active_controllers(self,
                    gpu=False):
        
        now_active = torch.nonzero(self._now_active.squeeze(dim=1)).squeeze(dim=1)
        if not now_active.shape[0] == 0:
            if gpu:
                return now_active.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return now_active    
        else:
            # no controller active
            return None
    
    def get_inactive_controllers(self,
                    gpu=False):
        
        not_active = torch.nonzero((~self._now_active).squeeze(dim=1)).squeeze(dim=1)
        if not not_active.shape[0] == 0:
            if gpu:
                return not_active.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return not_active
        else:
            # no controller active
            return None

    def get_failed_controllers(self,
                    gpu=False):
        
        failed = torch.nonzero(self._failed.squeeze(dim=1)).squeeze(dim=1)
        if not failed.shape[0] == 0:
            if gpu:
                return failed.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return failed
        else:
            # no controller active
            return None
    
    def get_registered_controllers(self,
                    gpu=False):

        registered = torch.nonzero(self._registered.squeeze(dim=1)).squeeze(dim=1)
        if not registered.shape[0] == 0:
            if gpu:
                return registered.cuda() # n_envs x 8 bits of CPU -> GPU (RX)
            else:
                return registered
        else:
            # no controller registered
            return None
        
    def just_started_running(self):

        return self.is_running() and (not self._was_running)
          
    def is_running(self):

        return self._is_running

    def _set_rhc_state(self):

        if self._using_gpu:
            # updates shared tensor on CPU with latest data from states on GPU
            # and writes to shared mem (GPU -> CPU copy here)
            # the total size of the data copied is
            # n_envs x (13 + 4*n_jnts + 6*n_contacts) * dtype_size 
            # with 16 joints + 4 contacts and float32 this gives
            # n_envs x 3232 bit -> with 160 env -> 0.51712MB
            # if the controllers runs at for example 0.03s
            # -> 17.24 MB/s of TX from GPU
            self._robot_states.synch_mirror(from_gpu=True)
        else:
            # only write to shared mem (gpu mirror is not used)
            self._robot_states.synch_to_shared_mem()

    def _get_rhc_sol(self):

        if self._using_gpu:
            self._rhc_cmds.synch_mirror(from_gpu=False) # read from shared mem and then copy to GPU
            # in a similar way to the rhc_state, this requires a copy, this time, from CPU to GPU (RX) of
            # n_envs x 3232 bit / update_dt
        else:
            # only read cmds from shared mem
            self._rhc_cmds.synch_from_shared_mem()

    def _require_pretrigger(self):
        if not self.pretriggered():
            exception = "Cannot call trigger() if pre_trigger()" + \
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
            
    def _check_running(self):
        if not self.is_running():
            exception = "Cluster server is not running! Did you call the run() method?"
            Journal.log(self.__class__.__name__,
                "_check_running",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)        

    def _pre_trigger_logs(self):
        
        self._rhc_status.controllers_counter.synch_all(retry = True,
                                                        read = True)
        n_controller = self._rhc_status.controllers_counter.get_torch_mirror()
        self._n_controllers_connected = n_controller[0, 0].item()

        if self._n_controllers_connected == 0:
            self._sporadic_log(calling_methd="trigger_solution",
                        msg = "waiting connection to ControlCluster server")
            
        if self._n_controllers_connected < self.cluster_size and \
                self._n_controllers_connected > 0 and \
                (self._trigger_counter+1) % self._print_frequency == 0:           
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
        
        active = self._rhc_status.activation_state.get_torch_mirror()
        if not active.all():
            msg = f"Controllers waiting to be activated... (" + \
                f"{self._rhc_status.activation_state.get_torch_mirror().sum().item()}/{self.cluster_size} active)"
            self._sporadic_log(calling_methd="trigger_solution",
                        msg = msg,
                        logtype=LogType.INFO)

    def _sporadic_log(self,
                calling_methd: str,
                msg: str,
                logtype: LogType = LogType.INFO):

        if self._verbose and \
            (self._trigger_counter+1) % self._print_frequency == 0: 
            
            Journal.log(self.__class__.__name__,
                calling_methd,
                msg,
                logtype,
                throw_when_excep = True)
                