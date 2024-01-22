from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype, toNumpyDType

from typing import Dict, Union, List
import numpy as np

# Simulator info

class SimData(SharedDataView):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0):

        basename = "SharedSimData" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_dims, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = True)

class DynamicSimInfoNames:

    def __init__(self):

        self._keys = ["sim_rt_factor", 
                "total_rt_factor",
                "env_stepping_dt",
                "task_prephysics_step_dt", 
                "world_stepping_dt",
                "cluster_state_update_dt",
                "cluster_sol_time",
                "OmniJntImpCntrl:time_to_update_state",
                "OmniJntImpCntrl:time_to_set_refs",
                "OmniJntImpCntrl:time_to_apply_cmds",
                "time_to_get_agent_data"]
        
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class SharedSimInfo:
                           
    def __init__(self, 
                is_server = False, 
                sim_params_dict: Dict = None):
        
        self.namespace = "SharedSimInfo"

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.sim_params_dict = sim_params_dict
        self._parse_sim_dict() # applies changes if needed

        self.param_keys = []

        self.dynamic_info = DynamicSimInfoNames()

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.dynamic_info.get() + list(self.sim_params_dict.keys())

        # actual data
        self.shared_sim_data = SimData(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = len(self.param_keys), 
                    verbose = True, 
                    vlevel = VLevel.V2)
        
        # names
        if self.is_server:

            self.shared_sim_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "SimDataNames", 
                                        name_space = self.namespace,
                                        verbose = True, 
                                        vlevel = VLevel.V1, 
                                        force_reconnection = True)

        else:

            self.shared_sim_datanames = StringTensorClient(
                                        basename = "SimDataNames", 
                                        name_space = self.namespace,
                                        verbose = True, 
                                        vlevel = VLevel.V1)
    
    def _parse_sim_dict(self):

        if self.sim_params_dict is not None:
        
            keys = list(self.sim_params_dict.keys())

            for key in keys:
                
                # we cannot mix types on a single entity of
                # shared memory

                if key == "gravity":
                    
                    # only vector param. supported (for now)

                    gravity = self.sim_params_dict[key]

                    self.sim_params_dict["g_x"] = gravity[0]
                    self.sim_params_dict["g_y"] = gravity[1]
                    self.sim_params_dict["g_z"] = gravity[2]

                    self.sim_params_dict.pop('gravity') # removes

                else:
        
                    if self.sim_params_dict[key] == "cpu":
        
                        self.sim_params_dict[key] = 0

                    if self.sim_params_dict[key] == "gpu" or \
                        self.sim_params_dict[key] == "cuda":
            
                        self.sim_params_dict[key] = 1
    
    def run(self):
        
        self.shared_sim_datanames.run()
        
        self.shared_sim_data.run()
            
        if self.is_server:
            
            names_written = self.shared_sim_datanames.write_vec(self.param_keys, 0)

            if not names_written:

                raise Exception("Could not write shared sim names on shared memory!")
            
            # writing static information to memory

        else:
            
            self.param_keys = [""] * self.shared_sim_datanames.length()

            names_read = self.shared_sim_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                raise Exception("Could not read shared sim names on shared memory!")
            
            self.shared_sim_data.synch_all(read=True, wait=True)
        
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(sharsor_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.sim_params_dict.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.dynamic_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.sim_params_dict[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_sim_data.write_wait(row_index=0,
                                    col_index=0,
                                    data=self.param_values)
                          
    def write(self,
            dyn_info_name: Union[str, List[str]],
            val: Union[float, List[float]]):

        # always writes to shared memory
        
        if isinstance(dyn_info_name, list):
            
            if not isinstance(val, list):

                raise Exception("The provided val should be a list of values!")
            
            if len(val) != len(dyn_info_name):

                raise Exception("Name list and values length mismatch!")

            for i in range(len(val)):
                
                idx = self.dynamic_info.get_idx(dyn_info_name[i])
                
                self.param_values[idx, 0] = val[i]
                
                self.shared_sim_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.dynamic_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_sim_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def synch(self):

        self.shared_sim_data.synch_all(read=True, wait = True)
    
    def get(self):

        self.synch()

        return self.shared_sim_data.numpy_view.copy()
    
    def close(self):

        self.shared_sim_data.close()
        self.shared_sim_datanames.close()

    def terminate(self):

        # just an alias for legacy compatibility
        self.close()

    def __del__(self):
        
        self.close()

# Control cluster

class ClusterCumulativeData(SharedDataView):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0):

        basename = "ClusterCumulativeData" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_dims, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = True)
        
class RtiSolTime(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True):

        basename = "RtiSolTime" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = cluster_size, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe)

class SolveLoopDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True):

        basename = "SolveLoopDt" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = cluster_size, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe)
     
class PrbUpdateDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True):

        basename = "PrbUpdateDt" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = cluster_size, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe)

class PhasesShiftDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True):

        basename = "PhasesShiftDt" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = cluster_size, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe)

class TaskRefUpdateDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True):

        basename = "TaskRefUpdateDt" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = cluster_size, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=sharsor_dtype.Float,
            fill_value=np.nan,
            safe = safe)
        
class DynamicClusterInfoNames:

    def __init__(self):

        self._keys = ["cluster_rt_factor", 
                "cluster_sol_time",
                "cluster_state_update_dt",
                "cluster_ready",
                "cluster_nominal_dt"]
        
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class ClusterStats:
                           
    def __init__(self, 
                cluster_size: int = 1,
                is_server = False, 
                param_dict: Dict = None,
                name = "",
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V2,
                safe: bool = True):
        
        self.cluster_size = cluster_size
        
        self.vlevel = vlevel
        self.verbose = verbose

        self.safe = safe

        self.name = name

        self.namespace = self.name + "ClusterStats"

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.param_dict = param_dict

        self.param_keys = []

        self.dynamic_info = DynamicClusterInfoNames()

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.dynamic_info.get() + list(self.param_dict.keys())

        self.shared_data = ClusterCumulativeData(namespace = self.namespace,
                            is_server = is_server, 
                            n_dims = len(self.param_keys),
                            verbose = verbose, 
                            vlevel = vlevel)

        self.rti_sol_time = RtiSolTime(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False)
        
        self.solve_loop_dt = SolveLoopDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False)
        
        self.prb_update_dt = PrbUpdateDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False)

        self.phase_shift_dt = PhasesShiftDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False)

        self.task_ref_update_dt = TaskRefUpdateDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False)
        
        # names
        if self.is_server:

            self.shared_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "DataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel, 
                                        force_reconnection = True)

        else:

            self.shared_datanames = StringTensorClient(
                                        basename = "DataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
    
    def __del__(self):

        self.close()
    
    def run(self):
        
        self.shared_datanames.run()
        
        self.shared_data.run()

        self.rti_sol_time.run()

        self.solve_loop_dt.run()

        self.prb_update_dt.run()

        self.phase_shift_dt.run()

        self.task_ref_update_dt.run()
            
        if self.is_server:
            
            names_written = self.shared_datanames.write_vec(self.param_keys, 0)

            if not names_written:

                raise Exception("Could not write shared sim names on shared memory!")
            
            # writing static information to memory
            
        else:
            
            self.param_keys = [""] * self.shared_datanames.length()

            names_read = self.shared_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                raise Exception("Could not read shared sim names on shared memory!")
            
            self.shared_data.synch_all(read=True, wait=True)
            
            # wait flag since safe = False doesn't do anything
            self.rti_sol_time.synch_all(read=True, wait=True)

            self.solve_loop_dt.synch_all(read=True, wait=True)

            self.cluster_size = self.rti_sol_time.n_rows
            
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(sharsor_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.param_dict.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.dynamic_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.param_dict[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_data.write_wait(row_index=0,
                                    col_index=0,
                                    data=self.param_values)
                          
    def write_info(self,
            dyn_info_name: Union[str, List[str]],
            val: Union[float, List[float]]):

        # always writes to shared memory
        
        if isinstance(dyn_info_name, list):
            
            if not isinstance(val, list):

                raise Exception("The provided val should be a list of values!")
            
            if len(val) != len(dyn_info_name):

                raise Exception("Name list and values length mismatch!")

            for i in range(len(val)):
                
                idx = self.dynamic_info.get_idx(dyn_info_name[i])
                
                self.param_values[idx, 0] = val[i]
                
                self.shared_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.dynamic_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def synch_info(self):

        self.shared_data.synch_all(read=True, wait = True)
    
    def synch_all(self):

        self.rti_sol_time.synch_all(read=True, wait = True)

        self.solve_loop_dt.synch_all(read=True, wait = True)

    def get_info(self):

        self.synch_info()

        return self.shared_data.numpy_view.copy()
    
    def close(self):

        self.shared_data.close()
        self.shared_datanames.close()

        self.rti_sol_time.close()
        self.solve_loop_dt.close()
        self.prb_update_dt.close()
        self.phase_shift_dt.close()
        self.task_ref_update_dt.close()

    def terminate(self):

        # just an alias for legacy compatibility
        self.close()
