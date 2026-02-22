# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of MPCHive and distributed under the General Public License version 2 license.
# 
# MPCHive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# MPCHive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MPCHive.  If not, see <http://www.gnu.org/licenses/>.
# 
from abc import ABC, abstractmethod
# from perf_sleep.pyperfsleep import PerfSleep
# from mpc_hive.utilities.cpu_utils.core_utils import get_memory_usage

import time 

import numpy as np

from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.shared_data.rhc_data import RhcCmds, RhcPred, RhcPredDelta
from mpc_hive.utilities.shared_data.rhc_data import RhcStatus
from mpc_hive.utilities.shared_data.rhc_data import RhcInternal
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling
from mpc_hive.utilities.remote_triggering import RemoteTriggererClnt

from mpc_hive.utilities.homing import RobotHomer

from mpc_hive.utilities.math_utils import world2base_frame

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import dtype

from typing import List
# from typing import TypeVar, Union

import signal
import os
import inspect

class RHController(ABC):

    def __init__(self, 
            srdf_path: str,
            n_nodes: int,
            dt: float,
            namespace: str, # shared mem namespace
            dtype = np.float32, 
            verbose = False, 
            debug = False,
            timeout_ms: int = 60000,
            allow_less_jnts: bool = True):
    
        signal.signal(signal.SIGINT, self._handle_sigint)

        self._allow_less_jnts = allow_less_jnts # whether to allow less joints in rhc controller than the ones on the robot
        # (e.g. some joints might not be desirable for control purposes)

        self.namespace = namespace
        self._dtype = dtype
        self._verbose = verbose
        self._debug = debug

        # if not self._debug:
        np.seterr(over='ignore') # ignoring overflows

        self._n_nodes = n_nodes
        self._dt = dt
        self._n_intervals = self._n_nodes - 1 
        self._t_horizon = self._n_intervals * dt
        self._set_rhc_pred_idx() # prection is by default written on last node
        self._set_rhc_cmds_idx() # default to idx 2 (i.e. cmds to get to third node)
        self.controller_index = None # will be assigned upon registration to a cluster
        self.controller_index_np = None 
        self._robot_mass=1.0

        self.srdf_path = srdf_path # using for parsing robot homing

        self._registered = False
        self._closed = False 
        
        self._allow_triggering_when_failed = True
        
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
        self.robot_pred = None
        self.rhc_pred_delta = None
        self.rhc_refs = None
        self._remote_triggerer = None
        self._remote_triggerer_timeout = timeout_ms # [ms]
        
        # remote termination
        self._remote_term = None
        self._term_req_received = False

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
        self._fail_idx_thresh = 5e3
        self._contact_f_thresh = 1e5

        self._failed = False

        self._start_time = time.perf_counter() # used for profiling when in debug mode

        self._homer = None # robot homing manager
        self._homer_env = None # used for setting homing to jnts not contained in rhc prb

        self._class_name_base = f"{self.__class__.__name__}"
        self._class_name = self._class_name_base # will append controller index upon registration

        self._contact_force_base_loc_aux=np.zeros((1,3),dtype=self._dtype)
        self._norm_grav_vector_w=np.zeros((1,3),dtype=self._dtype)
        self._norm_grav_vector_w[:, 2]=-1.0
        self._norm_grav_vector_base_loc=np.zeros((1,3),dtype=self._dtype)

        self._init() # initialize controller

        if not hasattr(self, '_rhc_fpaths'):
            self._rhc_fpaths = []
        self._rhc_fpaths.append(os.path.abspath(__file__))
        
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Get the file path of the class being initialized and append it to the list
        if not hasattr(cls, '_rhc_fpaths'):
            cls._rhc_fpaths = []
        child_class_file_path = os.path.abspath(inspect.getfile(cls))
        cls._rhc_fpaths.append(child_class_file_path)

    def this_paths(self):
        return self._rhc_fpaths
    
    def __del__(self):
        self._close()

    def _handle_sigint(self, signum, frame):
        if self._verbose:
            Journal.log(self._class_name,
                    "_handle_sigint",
                    "SIGINT received",
                    LogType.WARN)
        self._term_req_received = True
    
    def _set_rhc_pred_idx(self):
        # default to last node
        self._pred_node_idx=self._n_nodes-1
    
    def _set_rhc_cmds_idx(self):
        # use index 2 by default to compensate for
        # the inevitable action delay 
        # (get_state, trigger sol -> +dt -> apply sol )
        # if we apply action from second node (idenx 1) 
        # that action will already be one dt old. We assume we were already
        # applying the right action to get to the state of node idx 1 and get the 
        # cmds for reaching the state at idx 1
        self._rhc_cmds_node_idx=2

    def _close(self):
        if not self._closed:
            self._unregister_from_cluster()
            if self.robot_cmds is not None:
                self.robot_cmds.close()
            if self.robot_pred is not None:
                self.robot_pred.close()
            if self.rhc_pred_delta is not None:
                self.rhc_pred_delta.close()
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
            if self._remote_term is not None:
                self._remote_term.close()
            self._closed = True

    def close(self):
        self._close()

    def get_file_paths(self):
        # can be overriden by child
        base_paths = []
        base_paths.append(self._this_path)
        return base_paths
    
    def init_rhc_task_cmds(self):
        
        self.rhc_refs = self._init_rhc_task_cmds()
        self.rhc_refs.reset()
        
    def _init_states(self):
        
        quat_remap = self._get_quat_remap()
        self.robot_state = RobotState(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                optimize_mem=True,
                                n_robots=1, # we just need the row corresponding to this controller
                                n_jnts=None, # got from server
                                n_contacts=None # got from server
                                ) 
        self.robot_state.run()
        self.robot_cmds = RhcCmds(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                optimize_mem=True,
                                n_robots=1, # we just need the row corresponding to this controller
                                n_jnts=None, # got from server
                                n_contacts=None # got from server
                                ) 
        self.robot_cmds.run()
        self.robot_pred = RhcPred(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                optimize_mem=True,
                                n_robots=1, # we just need the row corresponding to this controller
                                n_jnts=None, # got from server
                                n_contacts=None # got from server
                                )
        self.robot_pred.run()
        self.rhc_pred_delta = RhcPredDelta(namespace=self.namespace,
                                is_server=False,
                                q_remapping=quat_remap, # remapping from environment to controller
                                with_gpu_mirror=False,
                                with_torch_view=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                optimize_mem=True,
                                n_robots=1, # we just need the row corresponding to this controller
                                n_jnts=None, # got from server
                                n_contacts=None # got from server
                                )
        self.rhc_pred_delta.run()

    def _rhc(self, rti: bool = True):
        if self._debug:
            self._rhc_db(rti=rti)
        else:
            self._rhc_min(rti=rti)
    
    def _rhc_db(self, rti: bool = True):
        # rhc with debug data
        self._start_time = time.perf_counter()

        self.robot_state.synch_from_shared_mem(robot_idx=self.controller_index, robot_idx_view=self.controller_index_np) # updates robot state with
        # latest data on shared mem

        self._compute_pred_delta()

        if not self.failed():
            # we can solve only if not in failure state
            if rti:
                self._failed = not self._solve() # solve actual TO with RTI
            else:
                self._failed = not self._bootstrap() # full bootstrap solve
            if (self._failed): 
                # perform failure procedure
                self._on_failure()                       
        else:
            if not self._allow_triggering_when_failed:
                Journal.log(self._class_name,
                    "solve",
                    f"Received solution req, but controller is in failure state. " + \
                        " You should have reset() the controller!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            else: 
                if self._verbose:
                    Journal.log(self._class_name,
                        "solve",
                        f"Received solution req, but controller is in failure state. No solution will be performed. " + \
                            " Use the reset() method to continue solving!",
                        LogType.WARN)
            
        self._write_cmds_from_sol() # we update update the views of the cmds
        # from the latest solution
    
        # in debug, rhc internal state is streamed over 
        # shared mem.
        self._update_rhc_internal()
        self._profiling_data_dict["full_solve_dt"] = time.perf_counter() - self._start_time
        self._update_profiling_data() # updates all profiling data
        if self._verbose:
            Journal.log(self._class_name,
                "solve",
                f"RHC full solve loop execution time  -> " + str(self._profiling_data_dict["full_solve_dt"]),
                LogType.INFO,
                throw_when_excep = True) 

    def _rhc_min(self, rti: bool = True):

        self.robot_state.synch_from_shared_mem(robot_idx=self.controller_index, robot_idx_view=self.controller_index_np) # updates robot state with
        # latest data on shared mem

        self._compute_pred_delta()

        if not self.failed():
            # we can solve only if not in failure state
            if rti:
                self._failed = not self._solve() # solve actual TO with RTI
            else:
                self._failed = not self._bootstrap() # full bootstrap solve
            if (self._failed):  
                # perform failure procedure
                self._on_failure()                       
        else:
            if not self._allow_triggering_when_failed:
                Journal.log(self._class_name,
                    "solve",
                    f"Received solution req, but controller is in failure state. " + \
                        " You should have reset() the controller!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            else: 
                if self._verbose:
                    Journal.log(self._class_name,
                        "solve",
                        f"Received solution req, but controller is in failure state. No solution will be performed. " + \
                            " Use the reset() method to continue solving!",
                        LogType.WARN)
                    
        self._write_cmds_from_sol() # we update the views of the cmds
        # from the latest solution even if failed
        
    def solve_once(self):
        # run a single iteration of the solve loop (used for pooling)
        if self._term_req_received:
            return False

        if not self._remote_triggerer.wait(self._remote_triggerer_timeout):
            Journal.log(self._class_name,
                f"solve",
                "Didn't receive any remote trigger req within timeout!",
                LogType.EXCEP,
                throw_when_excep = False)
            return False

        self._received_trigger = True

        if self.rhc_status.resets.read_retry(row_index=self.controller_index,
                                col_index=0,
                                row_index_view=0)[0]:
            self.reset() # rhc is reset

        if self.rhc_status.trigger.read_retry(row_index=self.controller_index,
                    col_index=0,
                    row_index_view=0)[0]:
            rti_solve = self.rhc_status.rti_solve.read_retry(row_index=self.controller_index,
                        col_index=0,
                        row_index_view=0)[0]
            self._rhc(rti=rti_solve) # run solution with requested solve mode
            self.rhc_status.trigger.write_retry(False, 
                row_index=self.controller_index,
                col_index=0,
                row_index_view=0) # allow next solution trigger 
        
        self._remote_triggerer.ack() # send ack signal to server
        self._received_trigger = False
        
        self._term_req_received = self._term_req_received or self._remote_term.read_retry(row_index=0,
                                                        col_index=0,
                                                        row_index_view=0)[0]
        
        if self._term_req_received:
            self.close()
            return False

        return True

    def solve(self):
        
        # run the solution loop and wait for trigger signals
        # using cond. variables (efficient)
        while True:
            if not self.solve_once():
                break
        
        self.close() # is not stricly necessary

    def reset(self):
        
        if not self._closed:
            self.reset_rhc_data()
            self._failed = False # allow triggering
            self._n_resets += 1
            self.rhc_status.fails.write_retry(False, 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)
            self.rhc_status.resets.write_retry(False, 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)

    def _create_jnt_maps(self):
        
        # retrieve env-side joint names from shared mem
        self._env_side_jnt_names = self.robot_state.jnt_names()
        self._check_jnt_names_compatibility() # will raise exception if not self._allow_less_jnts
        if not self._got_jnt_names_from_controllers:
            exception = f"Cannot run the solve(). assign_env_side_jnt_names() was not called!"
            Journal.log(self._class_name,
                    "_create_jnt_maps",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

        # robot homer guarantees that _controller_side_jnt_names is at least contained in self._env_side_jnt_names (or equal)
        self._to_controller = [self._env_side_jnt_names.index(element) for element in self._controller_side_jnt_names]
        # set joint remappings for shared data from env data to controller
        # all shared data is by convention specified according to the env (jnts are ordered that way)
        # the remapping is used so that when data is returned, its a remapped view from env to controller,
        # so that data can be assigned direclty from the rhc controller without any issues
        self.robot_state.set_jnts_remapping(jnts_remapping=self._to_controller)
        self.robot_cmds.set_jnts_remapping(jnts_remapping=self._to_controller)
        self.robot_pred.set_jnts_remapping(jnts_remapping=self._to_controller)
        self.rhc_pred_delta.set_jnts_remapping(jnts_remapping=self._to_controller)

        return True

    def reset_rhc_data(self):
        
        self._reset() # custom reset (e.g. it should set the current solution to some
        # default solution, like a bootstrap)

        self.rhc_refs.reset() # reset rhc refs to default (has to be called after _reset)

        self._write_cmds_from_sol() # use latest solution (e.g. from bootstrap if called before running
        # the first solve) as default state
    
    def failed(self):
        return self._failed

    def robot_mass(self):
        return self._robot_mass
    
    def _assign_cntrl_index(self, reg_state: np.ndarray):
        state = reg_state.flatten() # ensure 1D tensor
        free_spots = np.nonzero(~state.flatten())[0]
        return free_spots[0].item()  # just return the first free spot
    
    def _register_to_cluster(self):
        
        self.rhc_status = RhcStatus(is_server=False,
            namespace=self.namespace, 
            verbose=self._verbose, 
            vlevel=VLevel.V2,
            with_torch_view=False, 
            with_gpu_mirror=False,
            optimize_mem=True,
            cluster_size=1, # we just need the row corresponding to this controller
            n_contacts=None, # we get this from server
            n_nodes=None # we get this from server
            )
        self.rhc_status.run() # rhc status (reg. flags, failure, tot cost, tot cnstrl viol, etc...)

        # acquire semaphores since we have to perform non-atomic operations
        # on the whole memory views
        self.rhc_status.registration.data_sem_acquire()
        self.rhc_status.controllers_counter.data_sem_acquire()
        self.rhc_status.controllers_counter.synch_all(retry = True,
                                                read = True)

        available_spots = self.rhc_status.cluster_size
        # from here on all pre registration ops can be done

        # incrementing cluster controllers counter
        controllers_counter = self.rhc_status.controllers_counter.get_numpy_mirror()
        if controllers_counter[0, 0] + 1 > available_spots: # no space left -> return 
            self.rhc_status.controllers_counter.data_sem_release()
            self.rhc_status.registration.data_sem_release()
            exception = "Cannot register to cluster. No space left " + \
                f"({controllers_counter[0, 0]} controllers already registered)"
            Journal.log(self._class_name,
                    "_register_to_cluster",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True) 
        
        # now we can register 

        # increment controllers counter
        controllers_counter += 1 
        self.controller_index = controllers_counter.item() -1 
        
        # actually register to cluster
        self.rhc_status.controllers_counter.synch_all(retry = True,
            read = False) # writes to shared mem (just one for all controllers (i.e n_rows = 1))
        
        # read current registration state
        self.rhc_status.registration.synch_all(retry = True,
                                        read = True,
                                        row_index=self.controller_index,
                                        row_index_view=0)
        registrations = self.rhc_status.registration.get_numpy_mirror()
        # self.controller_index = self._assign_cntrl_index(registrations)
        

        self._class_name_base = self._class_name_base+str(self.controller_index)
        # self.controller_index_np = np.array(self.controller_index)
        self.controller_index_np = np.array(0) # given that we use optimize_mem, the shared mem copy has shape 1 x n_cols (we can write and read at [0, :])
        
        registrations[self.controller_index_np, 0] = True
        self.rhc_status.registration.synch_all(retry = True,
                                        read = False,
                                        row_index=self.controller_index,
                                        col_index=0,
                                        row_index_view=0) 

        # now all heavy stuff that would otherwise make the registration slow
        self._remote_term = SharedTWrapper(namespace=self.namespace,
            basename="RemoteTermination",
            is_server=False,
            verbose = self._verbose, 
            vlevel = VLevel.V2,
            with_gpu_mirror=False,
            with_torch_view=False,
            dtype=dtype.Bool)
        self._remote_term.run()

        # other initializations
        
        self.cluster_stats = RhcProfiling(is_server=False, 
                                    name=self.namespace,
                                    verbose=self._verbose,
                                    vlevel=VLevel.V2,
                                    safe=True,
                                    optimize_mem=True,
                                    cluster_size=1 # we just need the row corresponding to this controller
                                    ) # profiling data
        self.cluster_stats.run()
        self.cluster_stats.synch_info()
    
        self._create_jnt_maps()
        self.init_rhc_task_cmds() # initializes rhc interface to external commands (defined by child class)
        self._consinstency_checks() # sanity checks

        if self._homer is None:
            self._init_robot_homer() # call this in case it wasn't called by child
    
        self._robot_mass = self._get_robot_mass() # uses child class implemented method
        self._contact_f_scale = self._get_robot_mass() * 9.81

        # writing some static info about this controller
        # self.rhc_status.rhc_static_info.synch_all(retry = True,
        #     read = True,
        #     row_index=self.controller_index,
        #     col_index=0) # first read current static info from other controllers (not necessary if optimize_mem==True)
        self.rhc_status.rhc_static_info.set(data=np.array(self._dt),
            data_type="dts",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        self.rhc_status.rhc_static_info.set(data=np.array(self._t_horizon),
            data_type="horizons",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        self.rhc_status.rhc_static_info.set(data=np.array(self._n_nodes),
            data_type="nnodes",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        # writing some static rhc info which is only available after problem init
        self.rhc_status.rhc_static_info.set(data=np.array(len(self._get_contacts())),
            data_type="ncontacts",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        self.rhc_status.rhc_static_info.set(data=np.array(self.robot_mass()),
            data_type="robot_mass",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        self.rhc_status.rhc_static_info.set(data=np.array(self._pred_node_idx),
            data_type="pred_node_idx",
            rhc_idxs=self.controller_index_np,
            gpu=False)
        
        self.rhc_status.rhc_static_info.synch_retry(row_index=self.controller_index, 
            col_index=0,
            row_index_view=0,
            n_rows=1, n_cols=self.rhc_status.rhc_static_info.n_cols,
            read=False)
        
        # we set homings also for joints which are not in the rhc homing map
        # since this is usually required on server side
    
        homing_full = self._homer_env.get_homing().reshape(1, 
                        self.robot_cmds.n_jnts())
    
        null_action = np.zeros((1, self.robot_cmds.n_jnts()), 
                        dtype=self._dtype)

        self.robot_cmds.jnts_state.set(data=homing_full, data_type="q", 
                            robot_idxs=self.controller_index_np,
                            no_remap=True)
        self.robot_cmds.jnts_state.set(data=null_action, data_type="v", 
                            robot_idxs=self.controller_index_np,
                            no_remap=True)
        self.robot_cmds.jnts_state.set(data=null_action, data_type="eff", 
                            robot_idxs=self.controller_index_np,
                            no_remap=True)
        
        # write all joints (including homing for env-only ones)
        self.robot_cmds.jnts_state.synch_retry(row_index=self.controller_index, col_index=0,
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False) # only write data corresponding to this controller
        
        self.reset() # reset controller
        self._n_resets=0
        
        # for last we create the trigger client
        self._remote_triggerer = RemoteTriggererClnt(namespace=self.namespace,
                                        verbose=self._verbose,
                                        vlevel=VLevel.V2) # remote triggering
        self._remote_triggerer.run()

        if self._debug:
            # internal solution is published on shared mem
            # we assume the user has made available the cost
            # and constraint data at this point (e.g. through
            # the solution of a bootstrap)
            cost_data = self._get_cost_info()
            constr_data = self._get_constr_info()
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

        self._class_name = self._class_name + f"-{self.controller_index}"

        Journal.log(self._class_name,
                    "_register_to_cluster",
                    f"controller registered",
                    LogType.STAT,
                    throw_when_excep = True)
        
        # we can now release everything so that other controllers can register
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
                                    col_index=0,
                                    row_index_view=0)
            self._deactivate()
            # decrementing controllers counter
            self.rhc_status.controllers_counter.synch_all(retry = True,
                                                    read = True)
            controllers_counter = self.rhc_status.controllers_counter.get_numpy_mirror()
            controllers_counter -= 1 
            self.rhc_status.controllers_counter.synch_all(retry = True,
                                                    read = False)
            Journal.log(self._class_name,
                    "_unregister_from_cluster",
                    "Done",
                    LogType.STAT,
                    throw_when_excep = True)
            # we can now release everything
            self.rhc_status.registration.data_sem_release()
            self.rhc_status.controllers_counter.data_sem_release()
            self._registered = False

    def _get_quat_remap(self):
        # to be overridden by child class if necessary
        return [0, 1, 2, 3]
    
    def _consinstency_checks(self):
        
        # check controller dt
        server_side_cluster_dt = self.cluster_stats.get_info(info_name="cluster_dt")
        if not (abs(server_side_cluster_dt - self._dt) < 1e-4):
            exception = f"Trying to initialize a controller with control dt {self._dt}, which" + \
                f"does not match the cluster control dt {server_side_cluster_dt}"
            Journal.log(self._class_name,
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
        # check contact names
        
        server_side_contact_names = set(self.robot_state.contact_names())
        control_side_contact_names = set(self._get_contacts())

        if (not server_side_contact_names == control_side_contact_names) and self._verbose:
            warn = f"Controller-side contact names do not match server-side names!" + \
                f"\nServer: {self.robot_state.contact_names()}\n Controller: {self._get_contacts()}"
            Journal.log(self._class_name,
                        "_consinstency_checks",
                        warn,
                        LogType.WARN,
                        throw_when_excep = True)
        if not len(self.robot_state.contact_names()) == len(self._get_contacts()):
            # at least, we need the n of contacts to match!
            exception = f"Controller-side n contacts {self._get_contacts()} do not match " + \
                f"server-side n contacts {len(self.robot_state.contact_names())}!"
            Journal.log(self._class_name,
                        "_consinstency_checks",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
    def _init(self):

        stat = f"Trying to initialize RHC controller " + \
            f"with dt: {self._dt} s, t_horizon: {self._t_horizon} s, n_intervals: {self._n_intervals}"
        Journal.log(self._class_name,
                    "_init",
                    stat,
                    LogType.STAT,
                    throw_when_excep = True)
        
        self._init_states() # initializes shared mem. states 

        self._init_problem() # we call the child's initialization method for the actual problem
        self._post_problem_init()

        self._register_to_cluster() # registers the controller to the cluster

        Journal.log(self._class_name,
                    "_init",
                    f"RHC controller initialized with cluster index {self.controller_index} on process {os.getpid()}",
                    LogType.STAT,
                    throw_when_excep = True)

    def _deactivate(self):
        # signal controller deactivation over shared mem
        self.rhc_status.activation_state.write_retry(False, 
                                row_index=self.controller_index,
                                col_index=0,
                                row_index_view=0)
        # also set cmds to homing for safety
        # self.reset_rhc_data()

    def _on_failure(self):
        
        self.rhc_status.fails.write_retry(True, 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)
        self._deactivate()
        self._n_fails += 1
        self.rhc_status.controllers_fail_counter.write_retry(self._n_fails,
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)

    def _init_robot_homer(self):
        self._homer = RobotHomer(srdf_path=self.srdf_path, 
                            jnt_names=self._controller_side_jnt_names,
                            verbose=self._verbose)
        
        self._homer_env = RobotHomer(srdf_path=self.srdf_path, 
                            jnt_names=self.robot_state.jnt_names(),
                            verbose=self._verbose)
        
    def _update_profiling_data(self):

        # updated debug data on shared memory
        # with the latest info available
        self.cluster_stats.solve_loop_dt.write_retry(self._profiling_data_dict["full_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0,
                                                            row_index_view=0)
        self.cluster_stats.rti_sol_time.write_retry(self._profiling_data_dict["rti_solve_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0,
                                                            row_index_view=0)
        self.cluster_stats.prb_update_dt.write_retry(self._profiling_data_dict["problem_update_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0,
                                                            row_index_view=0)
        self.cluster_stats.phase_shift_dt.write_retry(self._profiling_data_dict["phases_shift_dt"], 
                                                            row_index=self.controller_index,
                                                            col_index=0,
                                                            row_index_view=0)
        self.cluster_stats.task_ref_update_dt.write_retry(self._profiling_data_dict["task_ref_update"], 
                                                            row_index=self.controller_index,
                                                            col_index=0,
                                                            row_index_view=0)
    
    def _write_cmds_from_sol(self):

        # gets data from the solution and updates the view on the shared data

        # cmds for robot
        # jnts
        self.robot_cmds.jnts_state.set(data=self._get_jnt_q_from_sol(node_idx=self._rhc_cmds_node_idx), data_type="q", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=self._get_jnt_v_from_sol(node_idx=self._rhc_cmds_node_idx), data_type="v", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=self._get_jnt_a_from_sol(node_idx=self._rhc_cmds_node_idx-1), data_type="a", robot_idxs=self.controller_index_np)
        self.robot_cmds.jnts_state.set(data=self._get_jnt_eff_from_sol(node_idx=self._rhc_cmds_node_idx-1), data_type="eff", robot_idxs=self.controller_index_np)
        # root
        self.robot_cmds.root_state.set(data=self._get_root_full_q_from_sol(node_idx=self._rhc_cmds_node_idx), data_type="q_full", robot_idxs=self.controller_index_np)
        self.robot_cmds.root_state.set(data=self._get_root_twist_from_sol(node_idx=self._rhc_cmds_node_idx), data_type="twist", robot_idxs=self.controller_index_np)
        self.robot_cmds.root_state.set(data=self._get_root_a_from_sol(node_idx=self._rhc_cmds_node_idx-1), data_type="a_full", robot_idxs=self.controller_index_np)
        self.robot_cmds.root_state.set(data=self._get_norm_grav_vector_from_sol(node_idx=self._rhc_cmds_node_idx-1), data_type="gn", robot_idxs=self.controller_index_np)
        f_contact = self._get_f_from_sol()
        if f_contact is not None:
            contact_names = self.robot_state.contact_names()
            node_idx_f_estimate=self._rhc_cmds_node_idx-1 # we always write the force to reach the desired state (prev node) 
            rhc_q_estimate=self._get_root_full_q_from_sol(node_idx=node_idx_f_estimate)[:, 3:7]
            for i in range(len(contact_names)):
                contact = contact_names[i]
                contact_idx = i*3
                contact_force_rhc_world=f_contact[contact_idx:(contact_idx+3), node_idx_f_estimate:node_idx_f_estimate+1].T
                world2base_frame(v_w=contact_force_rhc_world, 
                    q_b=rhc_q_estimate, 
                    v_out=self._contact_force_base_loc_aux,
                    is_q_wijk=False # horizon q is ijkw
                    )
                self.robot_cmds.contact_wrenches.set(data=self._contact_force_base_loc_aux, 
                    data_type="f", 
                    robot_idxs=self.controller_index_np,
                    contact_name=contact)
        
        # prediction data from MPC horizon
        self.robot_pred.jnts_state.set(data=self._get_jnt_q_from_sol(node_idx=self._pred_node_idx), data_type="q", robot_idxs=self.controller_index_np)
        self.robot_pred.jnts_state.set(data=self._get_jnt_v_from_sol(node_idx=self._pred_node_idx), data_type="v", robot_idxs=self.controller_index_np)
        self.robot_pred.jnts_state.set(data=self._get_jnt_a_from_sol(node_idx=self._pred_node_idx-1), data_type="a", robot_idxs=self.controller_index_np)
        self.robot_pred.jnts_state.set(data=self._get_jnt_eff_from_sol(node_idx=self._pred_node_idx-1), data_type="eff", robot_idxs=self.controller_index_np)
        self.robot_pred.root_state.set(data=self._get_root_full_q_from_sol(node_idx=self._pred_node_idx), data_type="q_full", robot_idxs=self.controller_index_np)
        self.robot_pred.root_state.set(data=self._get_root_twist_from_sol(node_idx=self._pred_node_idx), data_type="twist", robot_idxs=self.controller_index_np)
        self.robot_pred.root_state.set(data=self._get_root_a_from_sol(node_idx=self._pred_node_idx-1), data_type="a_full", robot_idxs=self.controller_index_np)
        self.robot_pred.root_state.set(data=self._get_norm_grav_vector_from_sol(node_idx=self._pred_node_idx-1), data_type="gn", robot_idxs=self.controller_index_np)

        # write robot commands
        self.robot_cmds.jnts_state.synch_retry(row_index=self.controller_index, col_index=0, 
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False) # jnt state
        self.robot_cmds.root_state.synch_retry(row_index=self.controller_index, col_index=0, 
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.root_state.n_cols,
                                read=False) # root state, in case it was written
        self.robot_cmds.contact_wrenches.synch_retry(row_index=self.controller_index, col_index=0, 
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.contact_wrenches.n_cols,
                                read=False) # contact state
        
        # write robot pred
        self.robot_pred.jnts_state.synch_retry(row_index=self.controller_index, col_index=0, 
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False)
        self.robot_pred.root_state.synch_retry(row_index=self.controller_index, col_index=0, 
                                row_index_view=0,
                                n_rows=1, n_cols=self.robot_cmds.root_state.n_cols,
                                read=False)
        
        # we also fill other data (cost, constr. violation, etc..)
        self.rhc_status.rhc_cost.write_retry(self._get_rhc_cost(), 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)
        self.rhc_status.rhc_constr_viol.write_retry(self._get_rhc_constr_viol(), 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)
        self.rhc_status.rhc_n_iter.write_retry(self._get_rhc_niter_to_sol(), 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0)
        self.rhc_status.rhc_nodes_cost.write_retry(data=self._get_rhc_nodes_cost(), 
                                    row_index=self.controller_index, 
                                    col_index=0,
                                    row_index_view=0)
        self.rhc_status.rhc_nodes_constr_viol.write_retry(data=self._get_rhc_nodes_constr_viol(), 
                                    row_index=self.controller_index, 
                                    col_index=0,
                                    row_index_view=0)
        self.rhc_status.rhc_fail_idx.write_retry(self._get_failure_index(), 
                                    row_index=self.controller_index,
                                    col_index=0,
                                    row_index_view=0) # write idx  on shared mem

    def _compute_pred_delta(self):
        
        # measurements
        q_full_root_meas = self.robot_state.root_state.get(data_type="q_full", robot_idxs=self.controller_index_np)
        twist_root_meas = self.robot_state.root_state.get(data_type="twist", robot_idxs=self.controller_index_np)
        a_root_meas = self.robot_state.root_state.get(data_type="a_full", robot_idxs=self.controller_index_np)
        g_vec_root_meas = self.robot_state.root_state.get(data_type="gn", robot_idxs=self.controller_index_np)

        q_jnts_meas = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index_np)
        v_jnts_meas = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index_np)
        a_jnts_meas = self.robot_state.jnts_state.get(data_type="a", robot_idxs=self.controller_index_np)
        eff_jnts_meas = self.robot_state.jnts_state.get(data_type="eff", robot_idxs=self.controller_index_np)

        # prediction from rhc 
        delta_root_q_full=self._get_root_full_q_from_sol(node_idx=1)-q_full_root_meas
        delta_root_twist=self._get_root_twist_from_sol(node_idx=1)-twist_root_meas
        delta_root_a=self._get_root_a_from_sol(node_idx=0)-a_root_meas
        delta_g_vec=self._get_norm_grav_vector_from_sol(node_idx=0)-g_vec_root_meas

        delta_jnts_q=self._get_jnt_q_from_sol(node_idx=1)-q_jnts_meas
        delta_jnts_v=self._get_jnt_v_from_sol(node_idx=1)-v_jnts_meas
        delta_jnts_a=self._get_jnt_a_from_sol(node_idx=0)-a_jnts_meas
        delta_jnts_eff=self._get_jnt_eff_from_sol(node_idx=0)-eff_jnts_meas

        # writing pred. errors
        self.rhc_pred_delta.root_state.set(data=delta_root_q_full, data_type="q_full", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.root_state.set(data=delta_root_twist,data_type="twist", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.root_state.set(data=delta_root_a,data_type="a_full", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.root_state.set(data=delta_g_vec, data_type="gn", robot_idxs=self.controller_index_np)

        self.rhc_pred_delta.jnts_state.set(data=delta_jnts_q,data_type="q", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.jnts_state.set(data=delta_jnts_v,data_type="v", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.jnts_state.set(data=delta_jnts_a,data_type="a", robot_idxs=self.controller_index_np)
        self.rhc_pred_delta.jnts_state.set(data=delta_jnts_eff, data_type="eff", robot_idxs=self.controller_index_np)

        # write on shared memory
        self.rhc_pred_delta.jnts_state.synch_retry(row_index=self.controller_index, 
                                                   col_index=0, 
                                                   n_rows=1, 
                                                   row_index_view=0,
                                                   n_cols=self.robot_cmds.jnts_state.n_cols,
                                read=False) # jnt state
        self.rhc_pred_delta.root_state.synch_retry(row_index=self.controller_index, 
                                                    col_index=0,
                                                    n_rows=1, 
                                                    row_index_view=0,
                                                    n_cols=self.robot_cmds.root_state.n_cols,
                                read=False) # root state
    
    def _assign_controller_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._controller_side_jnt_names = jnt_names
        self._got_jnt_names_from_controllers = True

    def _check_jnt_names_compatibility(self):

        set_rhc = set(self._controller_side_jnt_names)
        set_env  = set(self._env_side_jnt_names)
        
        if not set_rhc == set_env:
            rhc_is_missing=set_env-set_rhc
            env_is_missing=set_rhc-set_env

            msg_type=LogType.WARN
            message=""
            if not len(rhc_is_missing)==0: # allowed
                message = "\nSome env-side joint names are missing on rhc client-side!\n" + \
                "RHC-SERVER-SIDE-> \n" + \
                " ".join(self._env_side_jnt_names) + "\n" +\
                "RHC-CLIENT-SIDE -> \n" + \
                " ".join(self._controller_side_jnt_names) + "\n" \
                "\MISSING -> \n" + \
                " ".join(list(rhc_is_missing)) + "\n"
                if not self._allow_less_jnts: # raise exception
                    msg_type=LogType.EXCEP

            if not len(env_is_missing)==0: # not allowed
                message = "\nSome rhc-side joint names are missing on rhc server-side!\n" + \
                "RHC-SERVER-SIDE-> \n" + \
                " ".join(self._env_side_jnt_names) + \
                "RHC-CLIENT-SIDE -> \n" + \
                " ".join(self._controller_side_jnt_names) + "\n" \
                "\nmissing -> \n" + \
                " ".join(list(env_is_missing))
                msg_type=LogType.EXCEP
            
            if msg_type==LogType.WARN and not self._verbose:
                return
            
            Journal.log(self._class_name,
                    "_check_jnt_names_compatibility",
                    message,
                    msg_type,
                    throw_when_excep = True)
    
    def _get_cost_info(self):
        # to be overridden by child class
        return None, None
    
    def _get_constr_info(self):
        # to be overridden by child class
        return None, None
    
    def _get_fail_idx(self):
        # to be overriden by parent
        return 0.0
    
    def _get_failure_index(self):
        fail_idx=self._get_fail_idx()/self._fail_idx_thresh
        if fail_idx>1.0:
            fail_idx=1.0
        return fail_idx
    
    def _check_rhc_failure(self):
        # we use fail idx viol to detect failures
        idx = self._get_failure_index()
        return idx>=1.0
    
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
    def _get_jnt_q_from_sol(self, node_idx=1) -> np.ndarray:
        pass

    @abstractmethod
    def _get_jnt_v_from_sol(self, node_idx=1) -> np.ndarray:
        pass
    
    @abstractmethod
    def _get_jnt_a_from_sol(self, node_idx=0) -> np.ndarray:
        pass

    @abstractmethod
    def _get_jnt_eff_from_sol(self, node_idx=0) -> np.ndarray:
        pass
    
    @abstractmethod
    def _get_root_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        pass
    
    @abstractmethod
    def _get_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        pass

    @abstractmethod
    def _get_root_twist_from_sol(self, node_idx=1) -> np.ndarray:
        pass
    
    @abstractmethod
    def _get_root_a_from_sol(self, node_idx=0) -> np.ndarray:
        pass

    def _get_norm_grav_vector_from_sol(self, node_idx=1) -> np.ndarray:
        rhc_q=self._get_root_full_q_from_sol(node_idx=node_idx)[:, 3:7]
        world2base_frame(v_w=self._norm_grav_vector_w,q_b=rhc_q,v_out=self._norm_grav_vector_base_loc,
            is_q_wijk=False)
        return self._norm_grav_vector_base_loc
    
    def _get_rhc_cost(self):
        # to be overridden
        return np.nan
    
    def _get_rhc_constr_viol(self):
        # to be overridden
        return np.nan

    def _get_rhc_nodes_cost(self):
        # to be overridden
        return np.zeros((1,self.rhc_status.n_nodes), dtype=self._dtype)
    
    def _get_rhc_nodes_constr_viol(self):
        # to be overridden
        return np.zeros((1,self.rhc_status.n_nodes), dtype=self._dtype)
    
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
    def _bootstrap(self) -> bool:
        pass
            
    @abstractmethod
    def _get_ndofs(self):
        pass
    
    abstractmethod
    def _get_robot_mass(self):
        pass

    @abstractmethod
    def _init_problem(self):
        # initialized horizon's TO problem
        pass

    @abstractmethod
    def _post_problem_init(self):
        pass
