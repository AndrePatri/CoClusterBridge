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
from control_cluster_bridge.utilities.remote_triggering import RemoteTriggererClnt

from control_cluster_bridge.utilities.homing import RobotHomer
from control_cluster_bridge.utilities.cpu_utils.core_utils import get_memory_usage

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

from typing import List, TypeVar, Union

import numpy as np

from perf_sleep.pyperfsleep import PerfSleep

class RHController(ABC):

    def __init__(self, 
            srdf_path: str,
            n_nodes: int,
            dt: float,
            namespace: str, # shared mem namespace
            dtype = np.float32, 
            verbose = False, 
            debug = False):
        
        self.namespace = namespace
        self._dtype = dtype
        self._verbose = verbose
        self._debug = debug

        self._n_nodes = n_nodes
        self._dt = dt
        self._n_intervals = self._n_nodes - 1 
        self._t_horizon = self._n_intervals * dt
        
        self.controller_index = None # will be assigned upon registration to a cluster
        self.controller_index_np = None 

        self.srdf_path = srdf_path # using for parsing robot homing

        self._registered = False
        self._closed = False 

        self._profiling_data_dict = {}
        self._profiling_data_dict["full_solve_dt"] = np.nan
        self._profiling_data_dict["rti_solve_dt"] = np.nan
        self._profiling_data_dict["problem_update_dt"] = np.nan
        self._profiling_data_dict["phases_shift_dt"] = np.nan
        self._profiling_data_dict["task_ref_update"] = np.nan
        
        self.n_dofs = None
        self.n_contacts = None
        
        # shared mem
        self.robot_state = None 
        self.rhc_status = None
        self.rhc_internal = None
        self.cluster_stats = None
        self.robot_cmds = None
        self.rhc_refs = None
        self._remote_triggerer = None
        self._remote_triggerer_timeout = 120000 # [ns]
        
        # jnt names
        self._env_side_jnt_names = []
        self._controller_side_jnt_names = []
        self._got_jnt_names_from_controllers = False

        # data maps
        self._to_controller = []
        self._quat_remap = [0, 1, 2, 3] # defaults to no remap (to be overridden)

        self._got_contact_names = False

        self._received_trigger = False # used for proper termination

        self._n_resets = 0
        self._n_fails = 0
        self._failed = False

        self._start_time = time.perf_counter() # used for profiling when in debug mode

        self._homer = None # robot homing manager

        self._init()

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
        if self._remote_triggerer is not None:
            self._remote_triggerer.close()
        self._closed = True

    def init_rhc_task_cmds(self):
        
        self.rhc_refs = self._init_rhc_task_cmds()
        
    def _init_states(self):
        
        quat_remap = self._get_quat_remap()
        self.robot_state = RobotState(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2) 
        self.robot_state.run()
        self.robot_cmds = RhcCmds(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2) 
        self.robot_cmds.run()
    
    def _rhc(self):
        if self._debug:
            self._rhc_db()
        else:
            self._rhc_min()
    
    def _rhc_db(self):
        # rhc with debug data
        if self._debug:
            self._start_time = time.perf_counter()

        self.robot_state.synch_from_shared_mem() # updates robot state with
        # latest data on shared mem
        
        if not self.failed():
            # we can solve only if not in failure state
            self._failed = not self._solve() # solve actual TO
            if (self._failed):  
                # perform failure procedure
                self._on_failure()                       
        else:
            Journal.log(self.__class__.__name__ + f"{self.controller_index}",
                "solve",
                "Received solution req, but controller is in failure state. " + \
                    " You should have reset the controller!",
                LogType.EXCEP,
                throw_when_excep = True)
            
        self._write_cmds_from_sol() # we update update the views of the cmds
        # from the latest solution
    
        if self._debug:
            # if in debug, rhc internal state is streamed over 
            # shared mem.
            self._update_rhc_internal()

        self.rhc_status.trigger.write_retry(False, 
                                row_index=self.controller_index,
                                col_index=0) # allow next solution trigger 
        
        if self._debug:
            self._profiling_data_dict["full_solve_dt"] = time.perf_counter() - self._start_time
            self._update_profiling_data() # updates all profiling data
        
        if self._debug and self._verbose:
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                "solve",
                f"RHC full solve loop execution time  -> " + str(self._profiling_data_dict["full_solve_dt"]),
                LogType.INFO,
                throw_when_excep = True)  

    def _rhc_min(self):

        self.robot_state.synch_from_shared_mem() # updates robot state with
        # latest data on shared mem
        if not self.failed():
            # we can solve only if not in failure state
            self._failed = not self._solve() # solve actual TO
            if (self._failed):  
                # perform failure procedure
                self._on_failure()                       
        else:
            Journal.log(self.__class__.__name__ + f"{self.controller_index}",
                "solve",
                "Received solution req, but controller is in failure state. " + \
                    " You should have reset the controller!",
                LogType.EXCEP,
                throw_when_excep = True)
        self._write_cmds_from_sol() # we update update the views of the cmds
        # from the latest solution
        self.rhc_status.trigger.write_retry(False, 
                                row_index=self.controller_index,
                                col_index=0) # allow next solution trigger
            
    def solve(self):
        
        # run the solution loop and wait for trigger signals
        # using cond. variables (efficient)
        while True:
            try: 
                # we are always listening for a trigger signal 
                if not self._remote_triggerer.wait(self._remote_triggerer_timeout):
                    Journal.log(self.__class__.__name__,
                        "solve",
                        "Didn't receive any remote trigger req within timeout!",
                        LogType.EXCEP,
                        throw_when_excep = True)
                self._received_trigger = True
                # signal received -> we process incoming requests
                # perform reset, if required
                if self.rhc_status.resets.read_retry(row_index=self.controller_index,
                                                col_index=0)[0]:
                    self.reset() # rhc is reset
                # check if a trigger request was received
                if self.rhc_status.trigger.read_retry(row_index=self.controller_index,
                            col_index=0)[0]:
                    self._rhc() # run solution
                self._remote_triggerer.ack() # send ack signal to server
                self._received_trigger = False
            except (KeyboardInterrupt):
                self._close()
                break
                
    def reset(self):
        
        if not self._closed:
            aaa = f"AAAAAAAAAAAAAA {self._n_resets}- n {self.controller_index}\n"
            self._reset()
            self._failed = False # allow triggering
            self.set_cmds_to_homing()
            self._n_resets += 1
            self.rhc_status.fails.write_retry(False, 
                                        row_index=self.controller_index,
                                        col_index=0)
            self.rhc_status.resets.write_retry(False, 
                                            row_index=self.controller_index,
                                            col_index=0)

    def _create_jnt_maps(self):
        
        # retrieve env-side joint names from shared mem
        self._env_side_jnt_names = self.robot_state.jnt_names()
        self._check_jnt_names_compatibility() # will raise exception
        if not self._got_jnt_names_from_controllers:
            exception = f"Cannot run the solve(). assign_env_side_jnt_names() was not called!"
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                    "_create_jnt_maps",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
        self._to_controller = [self._env_side_jnt_names.index(element) for element in self._controller_side_jnt_names]
        # set joint remappings for shared data
        self.robot_state.set_jnts_remapping(jnts_remapping=self._to_controller)
        self.robot_cmds.set_jnts_remapping(jnts_remapping=self._to_controller)

        return True

    def set_cmds_to_homing(self):

        homing = self._homer.get_homing().reshape(1, 
                            self.robot_cmds.n_jnts())
        
        null_action = np.zeros((1, self.robot_cmds.n_jnts()), 
                        dtype=self._dtype)
        
        self.robot_cmds.jnts_state.set(data=homing, data_type="q", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=null_action, data_type="v", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=null_action, data_type="eff", robot_idxs=self.controller_index_np)

        self.robot_cmds.jnts_state.synch_retry(row_index=self.controller_index, col_index=0, n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False) # only write data corresponding to this controller
    
    def failed(self):
        return self._failed

    def _assign_cntrl_index(self, reg_state: np.ndarray):
        state = reg_state.flatten() # ensure 1D tensor
        free_spots = np.nonzero(~state.flatten())[0]
        return free_spots[0].item()  # just return the first free spot
    
    def _register_to_cluster(self):
        
        # acquire semaphores since we have to perform non-atomic operations
        # on the whole memory views
        self.rhc_status.registration.data_sem_acquire()
        self.rhc_status.controllers_counter.data_sem_acquire()
        self.rhc_status.controllers_counter.synch_all(retry = True,
                                                read = True)
        
        available_spots = self.rhc_status.cluster_size

        # incrementing cluster controllers counter
        controllers_counter = self.rhc_status.controllers_counter.get_numpy_view()
        if controllers_counter[0, 0] + 1 > available_spots: # no space left -> return 
            self.rhc_status.controllers_counter.data_sem_release()
            self.rhc_status.registration.data_sem_release()
            exception = "Cannot register to cluster. No space left " + \
                f"({controllers_counter[0, 0]} controllers already registered)"
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                    "_register_to_cluster",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True) 
                    
        # increment controllers counter
        controllers_counter += 1 
        self.rhc_status.controllers_counter.synch_all(retry = True,
                                                read = False) # writes to shared mem
        
        # read current registration state
        self.rhc_status.registration.synch_all(retry = True,
                                                read = True)
        registrations = self.rhc_status.registration.get_numpy_view()
        self.controller_index = self._assign_cntrl_index(registrations)
        self.controller_index_np = np.array(self.controller_index)

        registrations[self.controller_index, 0] = True
        self.rhc_status.registration.synch_all(retry = True,
                                read = False) # register

        Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                    "_register_to_cluster",
                    "Done",
                    LogType.STAT,
                    throw_when_excep = True)

        # we can now release everything
        self.rhc_status.controllers_counter.data_sem_release()
        self.rhc_status.registration.data_sem_release()
        
        self._registered = True
                              
    def _unregister_from_cluster(self):
        
        if self._received_trigger:
            # received interrupt during solution --> 
            # send ack signal to server anyway
            self._remote_triggerer.ack() 
        if self._registered:
            # acquire semaphores since we have to perform operations
            # on the whole memory views
            self.rhc_status.registration.data_sem_acquire()
            self.rhc_status.controllers_counter.data_sem_acquire()
            self.rhc_status.registration.write_retry(False, 
                                    row_index=self.controller_index,
                                    col_index=0)
            self._deactivate()
            # decrementing controllers counter
            self.rhc_status.controllers_counter.synch_all(retry = True,
                                                    read = True)
            controllers_counter = self.rhc_status.controllers_counter.get_numpy_view()
            controllers_counter -= 1 
            self.rhc_status.controllers_counter.synch_all(retry = True,
                                                    read = False)
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                    "_unregister_from_cluster",
                    "Done",
                    LogType.STAT,
                    throw_when_excep = True)
            # we can now release everything
            self.rhc_status.registration.data_sem_release()
            self.rhc_status.controllers_counter.data_sem_release()
    
    def _deactivate(self):
        # signal controller deactivation over shared mem
        self.rhc_status.activation_state.write_retry(False, 
                                row_index=self.controller_index,
                                col_index=0)
    
    def _get_quat_remap(self):
        # to be overridden by child class if necessary
        return [0, 1, 2, 3]
    
    def _consinstency_checks(self):
        
        # check controller dt
        server_side_cluster_dt = self.cluster_stats.get_info(info_name="cluster_dt")
        if not (abs(server_side_cluster_dt - self._dt) < 1e-8):
            exception = f"Trying to initialize a controller with control dt {self._dt}, which" + \
                f"does not match the cluster control dt {server_side_cluster_dt}"
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
        # check contact names
        
        server_side_contact_names = set(self.robot_state.contact_names())
        control_side_contact_names = set(self._get_contacts())

        if not server_side_contact_names == control_side_contact_names:
            warn = f"Controller-side contact names do not match server-side joint names!" + \
                f"\nServer: {self.robot_state.contact_names()}\n Controller: {self._get_contacts()}"
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                        "_consinstency_checks",
                        warn,
                        LogType.WARN,
                        throw_when_excep = True)
        if not len(self.robot_state.contact_names()) == len(self._get_contacts()):
            # at least, we need the n of contacts to match!
            exception = f"Controller-side n contacts {self._get_contacts()} do not match " + \
                f"server-side n contacts {len(self.robot_state.contact_names())}!"
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
    def _init(self):

        stat = f"Initializing RHC controller " + \
            f"with dt: {self._dt} s, t_horizon: {self._t_horizon} s, n_intervals: {self._n_intervals}"
        Journal.log(f"{self.__class__.__name__}",
                    "_init",
                    stat,
                    LogType.STAT,
                    throw_when_excep = True)
        
        self.rhc_status = RhcStatus(is_server=False,
                                    namespace=self.namespace, 
                                    verbose=self._verbose, 
                                    vlevel=VLevel.V2,
                                    with_torch_view=False, 
                                    with_gpu_mirror=False)
        self.rhc_status.run() # rhc status (reg. flags, failure, tot cost, tot cnstrl viol, etc...)
        self._register_to_cluster() # registers the controller to the cluster
        self._init_states() # initializes shared mem. states
        self._remote_triggerer = RemoteTriggererClnt(namespace=self.namespace,
                                        verbose=self._verbose,
                                        vlevel=VLevel.V2) # remote triggering
        self._remote_triggerer.run()
        self.cluster_stats = RhcProfiling(is_server=False, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2,
                                    safe=True) # profiling data
        self.cluster_stats.run()
        self.cluster_stats.synch_info()
        self._init_problem() # we call the child's initialization method for the actual problem
        self._create_jnt_maps()
        self.init_rhc_task_cmds() # initializes rhc interface to external commands (defined by child class)
        self._consinstency_checks() # sanity checks

        if self._debug:
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

        if self._homer is None:
            self._init_robot_homer() # call this in case it wasn't called by child
        self.set_cmds_to_homing()

        Journal.log(f"{self.__class__.__name__}",
                    "_init",
                    f"RHC controller initialized with index {self.controller_index}",
                    LogType.STAT,
                    throw_when_excep = True)

    def _on_failure(self):
        
        self.rhc_status.fails.write_retry(True, 
                                        row_index=self.controller_index,
                                        col_index=0)
        self._deactivate()
        self._n_fails += 1
        self.rhc_status.controllers_fail_counter.write_retry(self._n_fails,
                                                    row_index=self.controller_index,
                                                    col_index=0)

    def _init_robot_homer(self):
        self._homer = RobotHomer(srdf_path=self.srdf_path, 
                            jnt_names_prb=self._controller_side_jnt_names)
        
    def _update_profiling_data(self):

        # updated debug data on shared memory
        # with the latest info available
        self.cluster_stats.solve_loop_dt.write_retry(self._profiling_data_dict["full_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        self.cluster_stats.rti_sol_time.write_retry(self._profiling_data_dict["rti_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        self.cluster_stats.prb_update_dt.write_retry(self._profiling_data_dict["problem_update_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        self.cluster_stats.phase_shift_dt.write_retry(self._profiling_data_dict["phases_shift_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
        self.cluster_stats.task_ref_update_dt.write_retry(self._profiling_data_dict["task_ref_update"], 
                                                            row_index=self.controller_index,
                                                            col_index=0)
       
    def _write_cmds_from_sol(self):

        # gets data from the solution and updates the view on the shared data

        self.robot_cmds.jnts_state.set(data=self._get_cmd_jnt_q_from_sol(), data_type="q", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=self._get_cmd_jnt_v_from_sol(), data_type="v", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=self._get_cmd_jnt_eff_from_sol(), data_type="eff", robot_idxs=self.controller_index_np)
        
        # write to shared mem
        self.robot_cmds.jnts_state.synch_retry(row_index=self.controller_index, col_index=0, n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False)
        
        # we also fill other data (cost, constr. violation, etc..)
        self.rhc_status.rhc_cost.write_retry(self._get_rhc_cost(), 
                                    row_index=self.controller_index,
                                    col_index=0)
        self.rhc_status.rhc_constr_viol.write_retry(self._get_rhc_residual(), 
                                    row_index=self.controller_index,
                                    col_index=0)
        self.rhc_status.rhc_n_iter.write_retry(self._get_rhc_niter_to_sol(), 
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
            exception = "Server-side and client-side joint names do not match!\n" + \
                "server side -> \n" + \
                " ".join(self._env_side_jnt_names) + \
                "\nclient side -> \n" + \
                " ".join(self._controller_side_jnt_names) 
            Journal.log(f"{self.__class__.__name__}{self.controller_index}",
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
        # written so overhead is minimal for non-enabled data
        self.rhc_internal.write_q(data= self._get_q_from_sol(),
                            retry=True)
        self.rhc_internal.write_v(data= self._get_v_from_sol(),
                            retry=True)
        self.rhc_internal.write_a(data= self._get_a_from_sol(),
                            retry=True)
        self.rhc_internal.write_a_dot(data= self._get_a_dot_from_sol(),
                            retry=True)
        self.rhc_internal.write_f(data= self._get_f_from_sol(),
                            retry=True)
        self.rhc_internal.write_f_dot(data= self._get_f_dot_from_sol(),
                            retry=True)
        self.rhc_internal.write_eff(data= self._get_eff_from_sol(),
                            retry=True)
        for cost_idx in range(self.rhc_internal.config.n_costs):
            # iterate over all costs and update all values
            cost_name = self.rhc_internal.config.cost_names[cost_idx]
            self.rhc_internal.write_cost(data= self._get_cost_from_sol(cost_name = cost_name),
                                cost_name = cost_name,
                                retry=True)
        for constr_idx in range(self.rhc_internal.config.n_constr):
            # iterate over all constraints and update all values
            constr_name = self.rhc_internal.config.constr_names[constr_idx]
            self.rhc_internal.write_constr(data= self._get_constr_from_sol(constr_name=constr_name),
                                constr_name = constr_name,
                                retry=True)
    
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
    def _get_cmd_jnt_q_from_sol(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_cmd_jnt_v_from_sol(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _get_cmd_jnt_eff_from_sol(self) -> np.ndarray:
        pass

    def _get_rhc_cost(self) -> np.ndarray:
        # to be overridden
        return np.nan
    
    def _get_rhc_residual(self) -> np.ndarray:
        # to be overridden
        return np.nan

    def _get_rhc_niter_to_sol(self) -> np.ndarray:
        # to be overridden
        return np.nan
    
    @abstractmethod
    def _update_open_loop(self):
        # updates rhc controller 
        # using the internal state 
        pass
    
    @abstractmethod
    def _update_closed_loop(self):
        # uses meas. from robot
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
