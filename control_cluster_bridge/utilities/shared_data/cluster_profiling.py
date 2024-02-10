from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype, toNumpyDType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import Dict, Union, List
import numpy as np

# Control cluster profiling data

class ClusterCumulativeData(SharedDataView):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False):

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
            safe = True,
            force_reconnection=force_reconnection)
        
class RtiSolTime(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True,
        force_reconnection: bool = False):

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
            safe = safe,
            force_reconnection=force_reconnection)

class SolveLoopDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True,
        force_reconnection: bool = False):

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
            safe = safe,
            force_reconnection=force_reconnection)
     
class PrbUpdateDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True,
        force_reconnection: bool = False):

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
            safe = safe,
            force_reconnection=force_reconnection)

class PhasesShiftDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True,
        force_reconnection: bool = False):

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
            safe = safe,
            force_reconnection=force_reconnection)

class TaskRefUpdateDt(SharedDataView):
                 
    def __init__(self,
        cluster_size: int, 
        namespace = "",
        is_server = False, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        safe: bool = True,
        force_reconnection: bool = False):

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
            safe = safe,
            force_reconnection=force_reconnection)
        
class ClusterRuntimeInfoNames:

    def __init__(self):

        self._keys = ["cluster_rt_factor", 
                "cluster_sol_time",
                "cluster_ready"]
        
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class RhcProfiling(SharedDataBase):
                           
    def __init__(self, 
                cluster_size: int = 1,
                is_server = False, 
                param_dict: Dict = None,
                name = "",
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V2,
                safe: bool = True,
                force_reconnection: bool = False):
        
        self.cluster_size = cluster_size
        
        self.vlevel = vlevel
        self.verbose = verbose

        self.safe = safe

        self.name = name

        self.namespace = self.name + "RhcProfiling"

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.param_keys = []

        self.runtime_info =  ClusterRuntimeInfoNames()

        self.static_param_dict = param_dict

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.runtime_info.get() + list(self.static_param_dict.keys())

        self.shared_data = ClusterCumulativeData(namespace = self.namespace,
                            is_server = is_server, 
                            n_dims = len(self.param_keys),
                            verbose = verbose, 
                            vlevel = vlevel,
                            force_reconnection=force_reconnection)

        self.rti_sol_time = RtiSolTime(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False,
                            force_reconnection=force_reconnection)
        
        self.solve_loop_dt = SolveLoopDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False,
                            force_reconnection=force_reconnection)
        
        self.prb_update_dt = PrbUpdateDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False,
                            force_reconnection=force_reconnection)

        self.phase_shift_dt = PhasesShiftDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False,
                            force_reconnection=force_reconnection)

        self.task_ref_update_dt = TaskRefUpdateDt(cluster_size= cluster_size, 
                            namespace = self.namespace,
                            is_server = is_server, 
                            verbose = verbose, 
                            vlevel = vlevel,
                            safe=False,
                            force_reconnection=force_reconnection)
        
        # names
        if self.is_server:

            self.shared_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "DataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel, 
                                        force_reconnection = force_reconnection)

        else:

            self.shared_datanames = StringTensorClient(
                                        basename = "DataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
        
        self._is_runnning = False
    
    def __del__(self):

        self.close()
    
    def is_running(self):

        return self._is_runnning
    
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

                exception = "Could not write shared sim names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                                        
        else:
            
            self.param_keys = [""] * self.shared_datanames.length()

            names_read = self.shared_datanames.read_vec(self.param_keys, 0)

            if not names_read:
                
                exception = "Could not read shared sim names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self.shared_data.synch_all(read=True, wait=True)
            
            # wait flag since safe = False doesn't do anything
            self.rti_sol_time.synch_all(read=True, wait=True)

            self.solve_loop_dt.synch_all(read=True, wait=True)

            self.cluster_size = self.rti_sol_time.n_rows
            
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(sharsor_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.static_param_dict.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.runtime_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.static_param_dict[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_data.write_wait(row_index=0,
                                    col_index=0,
                                    data=self.param_values)

        self._is_runnning = True
                          
    def write_info(self,
            dyn_info_name: Union[str, List[str]],
            val: Union[float, List[float]]):

        # always writes to shared memory
        
        if isinstance(dyn_info_name, list):
            
            if not isinstance(val, list):
                
                exception = "The provided val should be a list of values!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
            if len(val) != len(dyn_info_name):

                exception = "Name list and values length mismatch!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            for i in range(len(val)):
                
                idx = self.runtime_info.get_idx(dyn_info_name[i])
                
                self.param_values[idx, 0] = val[i]
                
                self.shared_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.runtime_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_data.write_wait(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def get_static_info_idx(self, name: str):

        return self.idx_dict[name]
    
    def get_info(self,
            info_name: Union[str, List[str]]):
        
        if isinstance(info_name, list):
            
            return_list = []

            for i in range(len(info_name)):
                
                try:
                    
                    return_list.append(self.param_values[idx, 0].item())

                except ValueError:

                    pass

            return return_list
        
        elif isinstance(info_name, str):
            
            try:

                idx = self.param_keys.index(info_name)
            
                return self.param_values[idx, 0].item()

            except ValueError:

                pass
        
        else:

            exception = "The provided info_name should be a list strings or a string!"

            Logger.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def synch_info(self):

        self.shared_data.synch_all(read=True, wait = True)

        self.param_values[:, :] = self.shared_data.numpy_view

    def synch_all(self):

        self.rti_sol_time.synch_all(read=True, wait = True)

        self.solve_loop_dt.synch_all(read=True, wait = True)

    def get_all_info(self):

        self.synch_info()

        return self.param_values
    
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
