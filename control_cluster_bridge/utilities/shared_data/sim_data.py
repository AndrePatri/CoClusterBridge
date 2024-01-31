from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype, toNumpyDType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import Dict, Union, List
import numpy as np

# Simulator info

class SimData(SharedDataView):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False):

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
            safe = True,
            force_reconnection=force_reconnection)

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
    
class SharedSimInfo(SharedDataBase):
                           
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
            
        self._is_running = False
    
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
    
    def is_running(self):

        return self._is_running
    
    def run(self):
        
        self.shared_sim_datanames.run()
        
        self.shared_sim_data.run()
            
        if self.is_server:
            
            names_written = self.shared_sim_datanames.write_vec(self.param_keys, 0)

            if not names_written:
                
                exception = "Could not write shared sim names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
        else:
            
            self.param_keys = [""] * self.shared_sim_datanames.length()

            names_read = self.shared_sim_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                exception = "Could not read shared sim names on shared memory!"

                Logger.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
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
            
        self._is_running = True
                          
    def write(self,
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