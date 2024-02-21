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
from abc import ABC, abstractmethod

import time 

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal
from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling

from control_cluster_bridge.utilities.homing import RobotHomer
from control_cluster_bridge.utilities.cpu_utils.core_utils import get_memory_usage

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

from typing import List, TypeVar

import torch
import numpy as np

from perf_sleep.pyperfsleep import PerfSleep

class CntrlCmd(ABC):

    pass

class RHCCmd(ABC):

    pass

RHCCmdChild = TypeVar('RHCCmdChild', bound='RHCCmd')

class RHController(ABC):

    def __init__(self, 
            cluster_size: int,
            srdf_path: str,
            n_nodes: int,
            dt: float,
            verbose = False, 
            debug = False,
            array_dtype = torch.float32, 
            namespace = "",
            debug_sol = False):
        
        self.namespace = namespace
        
        self.perf_timer = PerfSleep()

        self.controller_index = None 
        self.controller_index_torch = None 

        self.srdf_path = srdf_path

        self._verbose = verbose
        self._debug = debug
        self._debug_sol = debug_sol

        self._profiling_data_dict = {}
        self._profiling_data_dict["full_solve_dt"] = np.nan
        self._profiling_data_dict["rti_solve_dt"] = np.nan
        self._profiling_data_dict["problem_update_dt"] = np.nan
        self._profiling_data_dict["phases_shift_dt"] = np.nan
        self._profiling_data_dict["task_ref_update"] = np.nan
        
        self.cluster_size = cluster_size
        self._registered = False
        self._closed = False 

        self.n_dofs = None
        self.n_contacts = None
        
        # shared mem
        self.robot_state = None 

        self.rhc_status = None
        self.rhc_internal = None
        self.cluster_stats = None

        self.robot_cmds = None
        self.rhc_refs = None

        # jnt names
        self._env_side_jnt_names = []
        self._controller_side_jnt_names = []

        self._got_jnt_names_from_controllers = False

        # data maps
        self._to_controller = []
        self._quat_remap = [0, 1, 2, 3] # defaults to no remap (to be overridden)
        self._jnt_maps_created = False
        
        self._states_initialized = False
        self._got_contact_names = False

        self._trigger_flag = False

        self.array_dtype = array_dtype

        self.add_data_lenght = 0

        self.n_resets = 0
        self.n_fails = 0
        self._failed = False

        self._n_nodes = n_nodes
        self._dt = dt
        self._n_intervals = self._n_nodes - 1 
        self._t_horizon = self._n_intervals * dt

        self._start_time = time.perf_counter()

        self._homer = None

        self._core_idx = None

        self._init()

    def init_rhc_task_cmds(self):
        
        self.rhc_refs = self._init_rhc_task_cmds()
        
    def init_states(self):
        
        quat_remap = self._get_quat_remap()

        self.robot_state = RobotState(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2) 
        self.robot_state.run()
        
        self.robot_cmds = RhcCmds(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2) 
        self.robot_cmds.run()
        
        self._states_initialized = True
    
    def _set_affinity(self):

        if self._core_idx is not None:
        
            import os

            pid = os.getpid()  

            os.sched_setaffinity(pid, {self._core_idx})

            info = f"Affinity ID {os.sched_getaffinity(pid)} was set for controller n.{self.controller_index}."

            Journal.log(self.__class__.__name__,
                        "_set_affinity",
                        info,
                        LogType.STAT,
                        throw_when_excep = True)

    def set_affinity(self, 
                core_idx: int):
        
        if not isinstance(core_idx, int):

            exception = f"core_idx should be an int"

            Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

        self._core_idx = core_idx

    def get_core_idx(self):

        return self._core_idx
    
    def solve(self):
        
        if not self._jnt_maps_created:
            
            exception = f"Jnt maps not initialized. Did you call the create_jnt_maps()?"

            Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
        if not self._states_initialized:

            exception =f"States not initialized. Did you call the init_states()?"

            Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

        if self.rhc_refs is None:

            exception = f"RHC task references non initialized. Did you call init_rhc_task_cmds()?"

            Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
        
        self._set_affinity() # set affinity, if core ids was provided

        while True:
            
            # we are always listening for a trigger signal from the client 
            # or a reset signal
            
            try:
                
                # print(f"{self.controller_index}-th memory usage: {get_memory_usage()} GB")
                
                # checks for reset requests
                if self.rhc_status.resets.read_wait(row_index=self.controller_index,
                                            col_index=0)[0]:

                    self.reset()

                # checks for trigger requests
                if self.rhc_status.trigger.read_wait(row_index=self.controller_index,
                                                    col_index=0)[0]:
                    
                    if self._debug:
                        
                        self._start_time = time.perf_counter()

                    self.robot_state.synch_from_shared_mem() # updates robot state with
                    # latest data on shared mem

                    # latest state is employed

                    if not self.failed():
                        
                        # we can solve only if not in failure state
                        self._failed = not self._solve() # solve actual TO

                    else:
                        
                        # perform failure procedure

                        self._on_failure()                       

                    self._write_cmds_from_sol() # we update update the views of the cmd
                    # from the latest solution

                    if self._debug_sol:
                        
                        # if in debug, rhc internal state is streamed over 
                        # shared mem.
                        self._update_rhc_internal()

                    # we signal the client this controller has finished its job by
                    # resetting the flag
                    self.rhc_status.trigger.write_wait(False, 
                                                    row_index=self.controller_index,
                                                    col_index=0)

                    if self._debug:
                        
                        self._profiling_data_dict["full_solve_dt"] = time.perf_counter() - self._start_time

                        self._update_profiling_data() # updates all profiling data

                    if self._verbose and self._debug:
                        
                        Journal.log(self.__class__.__name__ + str(self.controller_index),
                            "solve",
                            f"RHC full solve loop execution time  -> " + str(self._profiling_data_dict["full_solve_dt"]),
                            LogType.INFO,
                            throw_when_excep = True)
                
                else:
                    
                    # we avoid busy waiting and CPU saturation by sleeping for a small amount of time

                    self.perf_timer.clock_sleep(1000000) # nanoseconds (actually resolution is much
                    # poorer)

            except KeyboardInterrupt:

                break
    
    def reset(self):
        
        if not self._closed:

            self._reset()

            self._failed = False # allow triggering
            
            self.set_cmds_to_homing()

            # self.n_fails = 0 # reset fail n

            self.n_resets += 1

            self.rhc_status.fails.write_wait(False, 
                                        row_index=self.controller_index,
                                        col_index=0)
            
            self.rhc_status.resets.write_wait(False, 
                                            row_index=self.controller_index,
                                            col_index=0)

    def create_jnt_maps(self):
        
        # retrieve env-side joint names from shared mem
        self._env_side_jnt_names = self.robot_state.jnt_names()

        self._check_jnt_names_compatibility() # will raise exception

        if not self._got_jnt_names_from_controllers:
            
            exception = f"Cannot run the solve(). assign_env_side_jnt_names() was not called!"

            Journal.log(self.__class__.__name__,
                    "create_jnt_maps",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
        
        self._to_controller = [self._env_side_jnt_names.index(element) for element in self._controller_side_jnt_names]
        
        # set joint remappings for shared data
        self.robot_state.set_jnts_remapping(jnts_remapping=self._to_controller)
        self.robot_cmds.set_jnts_remapping(jnts_remapping=self._to_controller)

        self._jnt_maps_created = True

    def set_cmds_to_homing(self):

        homing = torch.tensor(self._homer.get_homing()).reshape(1, 
                            self.robot_cmds.n_jnts())
        
        null_action = torch.zeros((1, self.robot_cmds.n_jnts()), 
                        dtype=self.array_dtype)
        
        self.robot_cmds.jnts_state.set_q(q = homing, robot_idxs=self.controller_index_torch)

        self.robot_cmds.jnts_state.set_v(v = null_action, robot_idxs=self.controller_index_torch)

        self.robot_cmds.jnts_state.set_eff(eff = null_action, robot_idxs=self.controller_index_torch)

        self.robot_cmds.jnts_state.synch_wait(row_index=self.controller_index, col_index=0, n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False)
    
    def failed(self):

        return self._failed
    
    def __del__(self):
        
        if not self._closed:

            self._close()

    def _close(self):

        self._unregister_from_cluster()

        if self.robot_cmds is not None:
            
            self.robot_cmds.close()
        
        if self.robot_state is not None:
            
            self.robot_state.close()
        
        if self.rhc_status is not None:
        
            self.rhc_status.close()
        
        if self.rhc_internal is not None:

            self.rhc_internal.close()

        if self.cluster_stats is not None:

            self.cluster_stats.close()
        
        self._closed = True

    def _assign_cntrl_index(self, reg_state: torch.Tensor):

        control_index = 0
        
        state = reg_state.flatten() # ensure 1D tensor

        free_spots = torch.nonzero(~state).flatten()

        control_index = free_spots[0].item() # just return the first free spot

        return control_index
    
    def _register_to_cluster(self):
        
        # self._acquire_reg_sem()

        available_spots = self.rhc_status.cluster_size

        # incrementing cluster controllers counter
        self.rhc_status.controllers_counter.synch_all(wait = True,
                                                read = True)
        
        if self.rhc_status.controllers_counter.torch_view[0, 0] + 1 > available_spots:

            exception = "Cannot register to cluster. No space left " + \
                f"({self.rhc_status.controllers_counter.torch_view[0, 0]} controllers already registered)"

            Journal.log(self.__class__.__name__,
                    "_init",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
            
            exit()
        
        # increment controllers counter
        self.rhc_status.controllers_counter.torch_view += 1 
        self.rhc_status.controllers_counter.synch_all(wait = True,
                                                read = False) # writes to shared mem
        
        # read current registration state
        self.rhc_status.registration.synch_all(wait = True,
                                                read = True)
        
        self.controller_index = self._assign_cntrl_index(self.rhc_status.registration.torch_view)

        self.controller_index_torch = torch.tensor(self.controller_index)

        self.rhc_status.registration.torch_view[self.controller_index, 0] = True
        self.rhc_status.registration.synch_all(wait = True,
                                                read = False) # register

        # self._release_reg_sem()

        self._registered = True

    def _deactivate(self):

        # signal controller deactivation over shared mem
            self.rhc_status.activation_state.write_wait(False, 
                                    row_index=self.controller_index,
                                    col_index=0)
                                    
    def _unregister_from_cluster(self):

        if self._registered:

            # self._acquire_reg_sem()

            self.rhc_status.registration.write_wait(False, 
                                    row_index=self.controller_index,
                                    col_index=0)
            
            self._deactivate()
            
            # decrementing controllers counter
            self.rhc_status.controllers_counter.synch_all(wait = True,
                                                    read = True)
            self.rhc_status.controllers_counter.torch_view -= 1 
            self.rhc_status.controllers_counter.synch_all(wait = True,
                                                    read = False)

            # self._release_reg_sem()
    
    def _acquire_reg_sem(self):

        while not self.rhc_status.acquire_reg_sem():

            warn = "Trying to acquire registration flags, but failed. Retrying.."

            Journal.log(self.__class__.__name__,
                    "_acquire_reg_sem",
                    warn,
                    LogType.WARN,
                    throw_when_excep = True)
            
            self.perf_timer.clock_sleep(1000000)
            
    def _release_reg_sem(self):

        if not self.rhc_status.release_reg_sem():
            
            exception = "Failed to release registration flags."

            Journal.log(self.__class__.__name__,
                    "_release_reg_sem",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
    
    def _get_quat_remap(self):

        # to be overridden

        return [0, 1, 2, 3]
    
    def _consinstency_checks(self):
        
        # check controller dt
        server_side_cluster_dt = self.cluster_stats.get_info(info_name="cluster_dt")
  
        if not (abs(server_side_cluster_dt - self._dt) < 1e-8):

            exception = f"Trying to initialize a controller with control dt {self._dt}, which" + \
                f"does not match the cluster control dt {server_side_cluster_dt}"
        
            Journal.log(self.__class__.__name__,
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
            exit()
        
        # check contact names
        
        server_side_contact_names = set(self.robot_state.contact_names())
        control_side_contact_names = set(self._get_contacts())

        if not server_side_contact_names == control_side_contact_names:

            warn = f"Controller-side contact names do not match server-side joint names!" + \
                f"\nServer: {self.robot_state.contact_names()}\n Controller: {self._get_contacts()}"
        
            Journal.log(self.__class__.__name__,
                        "_consinstency_checks",
                        warn,
                        LogType.WARN,
                        throw_when_excep = True)
        
        if not len(self.robot_state.contact_names()) == len(self._get_contacts()):

            exception = f"Controller-side n contacts {self._get_contacts()} do not match " + \
                f"server-side n contacts {len(self.robot_state.contact_names())}!"
        
            Journal.log(self.__class__.__name__,
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            exit()
            
    def _init(self):

        stat = f"Initializing RHC controller " + \
            f"with dt: {self._dt} s, t_horizon: {self._t_horizon} s, n_intervals: {self._n_intervals}"
        
        Journal.log(self.__class__.__name__,
                    "_init",
                    stat,
                    LogType.STAT,
                    throw_when_excep = True)
        
        self._init_problem() # we call the child's initialization method

        self.rhc_status = RhcStatus(is_server=False,
                                    namespace=self.namespace, 
                                    verbose=True, 
                                    vlevel=VLevel.V2)

        self.rhc_status.run()

        # statistical data

        self.cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                    is_server=False, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2,
                                    safe=True)

        self.cluster_stats.run()
        self.cluster_stats.synch_info()

        self._register_to_cluster() # registers the controller to the cluster

        self.init_states() # initializes states

        self.create_jnt_maps()

        self.init_rhc_task_cmds() # initializes rhc commands
        
        self._consinstency_checks()

        if self._debug_sol:
            
            # internal solution is published on shared mem

            # we assume the user has made available the cost
            # and constraint data at this point (e.g. through
            # the solution of a bootstrap)
            
            cost_data = self._get_cost_data()
            constr_data = self._get_constr_data()

            config = RhcInternal.Config(is_server=True, 
                        enable_q= True, 
                        enable_v=True, 
                        enable_a=True, 
                        enable_a_dot=False, 
                        enable_f=True,
                        enable_f_dot=False, 
                        enable_eff=False, 
                        cost_names=cost_data[0], 
                        cost_dims=cost_data[1],
                        constr_names=constr_data[0],
                        constr_dims=constr_data[1],
                        )
            
            self.rhc_internal = RhcInternal(config=config, 
                                    namespace=self.namespace,
                                    rhc_index = self.controller_index,
                                    n_contacts=self.n_contacts,
                                    n_jnts=self.n_dofs,
                                    jnt_names=self._controller_side_jnt_names,
                                    n_nodes=self._n_nodes,
                                    verbose = self._verbose,
                                    vlevel=VLevel.V2,
                                    force_reconnection=True,
                                    safe=True)
            
            self.rhc_internal.run()

        self._homer = RobotHomer(self.srdf_path, 
                            self._controller_side_jnt_names)

        if self._homer is None:

            exception = f"Robot homer not initialized. Did you call the _init_robot_homer() method in the child class?"

            Journal.log(self.__class__.__name__,
                    "create_jnt_maps",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
        self.set_cmds_to_homing()

        Journal.log(self.__class__.__name__,
                    "_init",
                    f"RHC controller initialized with index {self.controller_index}",
                    LogType.STAT,
                    throw_when_excep = True)

    def _on_failure(self):
        
        self.rhc_status.fails.write_wait(True, 
                                        row_index=self.controller_index,
                                        col_index=0)
        
        self._deactivate()

        self.n_fails += 1

        self.rhc_status.controllers_fail_counter.write_wait(self.n_fails,
                                                    row_index=self.controller_index,
                                                    col_index=0)

    def _init_robot_homer(self):

        self._homer = RobotHomer(srdf_path=self.srdf_path, 
                            jnt_names_prb=self._get_robot_jnt_names())
        
    def _update_profiling_data(self):

        # updated debug data on shared memory
        # with the latest info available
        self.cluster_stats.solve_loop_dt.write_wait(self._profiling_data_dict["full_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        
        self.cluster_stats.rti_sol_time.write_wait(self._profiling_data_dict["rti_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        
        self.cluster_stats.prb_update_dt.write_wait(self._profiling_data_dict["problem_update_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        
        self.cluster_stats.phase_shift_dt.write_wait(self._profiling_data_dict["phases_shift_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        
        self.cluster_stats.task_ref_update_dt.write_wait(self._profiling_data_dict["task_ref_update"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
       
    def _write_cmds_from_sol(self):

        # gets data from the solution and updates the view on the shared data

        self.robot_cmds.jnts_state.set_q(q = self._get_cmd_jnt_q_from_sol(), robot_idxs=self.controller_index_torch)

        self.robot_cmds.jnts_state.set_v(v = self._get_cmd_jnt_v_from_sol(), robot_idxs=self.controller_index_torch)

        self.robot_cmds.jnts_state.set_eff(eff = self._get_cmd_jnt_eff_from_sol(), robot_idxs=self.controller_index_torch)
        
        # write to shared mem
        self.robot_cmds.jnts_state.synch_wait(row_index=self.controller_index, col_index=0, n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False)
        
        # we also fill other data (cost, constr. violation, etc..)
        self.rhc_status.rhc_cost.write_wait(self._get_rhc_cost(), 
                                    row_index=self.controller_index,
                                    col_index=0)
        self.rhc_status.rhc_constr_viol.write_wait(self._get_rhc_residual(), 
                                    row_index=self.controller_index,
                                    col_index=0)
        self.rhc_status.rhc_n_iter.write_wait(self._get_rhc_niter_to_sol(), 
                                    row_index=self.controller_index,
                                    col_index=0)
    
    def _assign_controller_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._controller_side_jnt_names = jnt_names

        self._got_jnt_names_from_controllers = True

    def _check_jnt_names_compatibility(self):

        set_srvr = set(self._controller_side_jnt_names)
        set_client  = set(self._env_side_jnt_names)

        if not set_srvr == set_client:

            exception = "Server-side and client-side joint names do not match!"

            Journal.log(self.__class__.__name__,
                    "_check_jnt_names_compatibility",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
    
    def _get_cost_data(self):
        
        # to be overridden by child class
        return None, None
    
    def _get_constr_data(self):
        
        # to be overridden by child class
        return None, None
    
    def _update_rhc_internal(self):

        # data which is not enabled in the config is not actually 
        # written so overhead is minimal in for non-enabled data

        self.rhc_internal.write_q(data= self._get_q_from_sol(),
                            wait=True)

        self.rhc_internal.write_v(data= self._get_v_from_sol(),
                            wait=True)
        
        self.rhc_internal.write_a(data= self._get_a_from_sol(),
                            wait=True)
        
        self.rhc_internal.write_a_dot(data= self._get_a_dot_from_sol(),
                            wait=True)
        
        self.rhc_internal.write_f(data= self._get_f_from_sol(),
                            wait=True)
        
        self.rhc_internal.write_f_dot(data= self._get_f_dot_from_sol(),
                            wait=True)
        
        self.rhc_internal.write_eff(data= self._get_eff_from_sol(),
                            wait=True)

        for cost_idx in range(self.rhc_internal.config.n_costs):
            
            # iterate over all costs and update all values
            cost_name = self.rhc_internal.config.cost_names[cost_idx]

            self.rhc_internal.write_cost(data= self._get_cost_from_sol(cost_name = cost_name),
                                cost_name = cost_name,
                                wait=True)
        
        for constr_idx in range(self.rhc_internal.config.n_constr):

            # iterate over all constraints and update all values
            constr_name = self.rhc_internal.config.constr_names[constr_idx]

            self.rhc_internal.write_constr(data= self._get_constr_from_sol(constr_name=constr_name),
                                constr_name = constr_name,
                                wait=True)
    
    def _get_contacts(self):
        
        contact_names = self._get_contact_names()

        self._got_contact_names = True

        return contact_names
    
    def _get_q_from_sol(self):

        # to be overridden by child class

        return None

    def _get_v_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_a_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_a_dot_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_f_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_f_dot_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_eff_from_sol(self):

        # to be overridden by child class
        
        return None
    
    def _get_cost_from_sol(self,
                    cost_name: str):

        # to be overridden by child class
        
        return None
    
    def _get_constr_from_sol(self,
                    constr_name: str):

        # to be overridden by child class
        
        return None
    
    @abstractmethod
    def _reset(self):
        
        pass

    @abstractmethod
    def _init_rhc_task_cmds(self):

        pass

    @abstractmethod
    def _get_robot_jnt_names(self):

        pass
    
    @abstractmethod
    def _get_contact_names(self):

        pass

    @abstractmethod
    def _get_cmd_jnt_q_from_sol(self) -> torch.Tensor:

        pass

    @abstractmethod
    def _get_cmd_jnt_v_from_sol(self) -> torch.Tensor:

        pass
    
    @abstractmethod
    def _get_cmd_jnt_eff_from_sol(self) -> torch.Tensor:

        pass

    def _get_rhc_cost(self) -> torch.Tensor:

        # to be overridden

        return np.nan
    
    def _get_rhc_residual(self) -> torch.Tensor:
        
        # to be overridden

        return np.nan

    def _get_rhc_niter_to_sol(self) -> torch.Tensor:
        
        # to be overridden
        
        return np.nan
    
    @abstractmethod
    def _update_open_loop(self):

        # updates measured robot state 
        # using the internal robot state of the RHC controller

        pass
    
    @abstractmethod
    def _update_closed_loop(self):

        # updates measured robot state 
        # using the provided measurements

        pass

    @abstractmethod
    def _solve(self) -> bool:

        pass
            
    @abstractmethod
    def _get_ndofs(self):

        pass

    @abstractmethod
    def _init_problem(self):

        # initialized horizon's TO problem

        pass
   
RHChild = TypeVar('RHChild', bound='RHController')