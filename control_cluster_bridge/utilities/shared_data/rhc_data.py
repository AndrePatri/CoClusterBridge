from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from control_cluster_bridge.utilities.shared_data.state_encoding import FullRobState
import numpy as np

import torch

from typing import List

# implementations for robot state and 
# commands from rhc controller
        
class RobotState(FullRobState):

    def __init__(self,
            namespace: str,
            is_server: bool,
            n_robots: int = None,
            n_jnts: int = None,
            n_contacts: int = 1,
            jnt_names: List[str] = None,
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            with_gpu_mirror: bool = True,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1):

        basename = "RobotState"

        super().__init__(namespace=namespace,
            basename=basename,
            is_server=is_server,
            n_robots=n_robots,
            n_jnts=n_jnts,
            n_contacts=n_contacts,
            jnt_names=jnt_names,
            contact_names=contact_names,
            q_remapping=q_remapping,
            with_gpu_mirror=with_gpu_mirror,
            force_reconnection=force_reconnection,
            safe=safe,
            verbose=verbose,
            vlevel=vlevel)

class RhcCmds(FullRobState):

    def __init__(self,
            namespace: str,
            is_server: bool,
            n_robots: int = None,
            n_jnts: int = None,
            n_contacts: int = 1,
            jnt_names: List[str] = None,
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            with_gpu_mirror: bool = True,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1):

        basename = "RhcCmds"

        super().__init__(namespace=namespace,
            basename=basename,
            is_server=is_server,
            n_robots=n_robots,
            n_jnts=n_jnts,
            n_contacts=n_contacts,
            jnt_names=jnt_names,
            contact_names=contact_names,
            q_remapping=q_remapping,
            with_gpu_mirror=with_gpu_mirror,
            force_reconnection=force_reconnection,
            safe=safe,
            verbose=verbose,
            vlevel=vlevel)

class RhcStatus(SharedDataBase):
    
    class FailFlagView(SharedDataView):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterFailFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomdic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
    
    class ResetFlagView(SharedDataView):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterResetFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
    
    class TriggerFlagView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterTriggerFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
    
    class ActivationFlagView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterActivationFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
    
    class RegistrationFlagView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterRegistrationFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
            
    class ControllersCounterView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ClusterControllersCounter" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = 1, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = True, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Int,
                force_reconnection=force_reconnection,
                fill_value = 0)

    class SemView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "SemView" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = 1, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = True, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Int,
                force_reconnection=force_reconnection,
                fill_value = 0)
            
    def __init__(self, 
            is_server = False, 
            cluster_size: int = -1, 
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        self.is_server = is_server

        self.cluster_size = cluster_size

        self.namespace = namespace

        self.verbose = verbose

        self.vlevel = vlevel

        self.fails = self.FailFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.resets = self.ResetFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.trigger = self.TriggerFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.activation_state = self.ActivationFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.registration = self.RegistrationFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)

        self.controllers_counter = self.ControllersCounterView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.registering_sem = self.ControllersCounterView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self.sem_view = self.SemView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection)
        
        self._is_runnning = False

        self._acquired_reg_sem = False
        
    def __del__(self):

        self.close()

    def is_running(self):
    
        return self._is_runnning
    
    def acquire_reg_sem(self):

        if self.is_running():
            
            if not self._acquired_reg_sem:

                self.sem_view.synch_all(read=True, wait=True)

                if self.sem_view.torch_view[0, 0] == 0:

                    self.sem_view.torch_view[0, 0] = 1 # acquire sem

                    self.sem_view.synch_all(read=False, wait=True)

                    self._acquired_reg_sem = True

            return self._acquired_reg_sem

    def release_reg_sem(self):

        if self.is_running():
                        
            if not self._acquired_reg_sem:

                return False
            
            else:

                self.sem_view.synch_all(read=True, wait=True)

                if self.sem_view.torch_view[0, 0] == 0:

                    exception = f"Reg. semaphore was acquired, but seems to be free on shared mem!"

                    Journal.log(self.__class__.__name__,
                        "run",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
                
                self.sem_view.torch_view[0, 0] = 0 # release sem

                self.sem_view.synch_all(read=False, wait=True)

                self._acquired_reg_sem = False

                return True

    def run(self):

        self.resets.run()
        self.trigger.run()
        self.fails.run()
        self.activation_state.run()
        self.registration.run()
        self.controllers_counter.run()
        self.sem_view.run()

        if not self.is_server:
    
            self.cluster_size = self.trigger.n_rows
        
        self._is_runnning = True

    def close(self):
        
        self.trigger.close()
        self.resets.close()
        self.fails.close()    
        self.activation_state.close()
        self.registration.close()
        self.controllers_counter.close()
        self.sem_view.close()

class RhcInternal(SharedDataBase):

    # class for sharing internal data of a 
    # receding-horizon controller

    class Q(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "q" # configuration vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
    
    class V(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "q" # velocity vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
    
    class A(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "A" # acceleration vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
    
    class ADot(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "a_dot" # jerk vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
    
    class F(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "f" # cartesian force vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
            
    class FDot(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "f_dot" # yank vector

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)
            
    class Eff(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_dims: int = -1, 
                n_nodes: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                fill_value: float = np.nan,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "v" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value = fill_value, 
                safe = safe,
                force_reconnection=force_reconnection)

    class RHCosts(SharedTensorDict):

        def __init__(self,
                names: List[str] = None, # not needed if client
                dimensions: List[int] = None, # not needed if client
                n_nodes: int = -1, # not needed if client 
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "rhc_costs"

            super().__init__(names = names, # not needed if client
                    dimensions = dimensions, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = namespace + basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe,
                    force_reconnection = force_reconnection) 
    
    class RHConstr(SharedTensorDict):

        def __init__(self,
                names: List[str] = None, # not needed if client
                dimensions: List[int] = None, # not needed if client
                n_nodes: int = -1, # not needed if client 
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False):
            
            basename = "rhc_constraints"

            super().__init__(names = names, # not needed if client
                    dimensions = dimensions, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = namespace + basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe,
                    force_reconnection = force_reconnection) 
    
    class Config():

        def __init__(self,
            is_server: bool = False,
            enable_q: bool = True, 
            enable_v: bool = True, 
            enable_a: bool = True,
            enable_a_dot: bool = False, 
            enable_f: bool = True, 
            enable_f_dot: bool = False, 
            enable_eff: bool = True, 
            cost_names: List[str] = None, 
            constr_names: List[str] = None,
            cost_dims: List[int] = None, 
            constr_dims: List[int] = None,
            enable_costs: bool = False,
            enable_constr: bool = False):

            self.is_server = is_server

            self.enable_q = enable_q
            self.enable_v = enable_v
            self.enable_a = enable_a
            self.enable_a_dot = enable_a_dot
            self.enable_f = enable_f
            self.enable_f_dot = enable_f_dot
            self.enable_eff = enable_eff

            self.enable_costs = enable_costs
            self.enable_constr = enable_constr

            self.cost_names = None
            self.cost_dims = None

            self.constr_names = None
            self.constr_dims = None

            self.n_costs = 0
            self.n_constr = 0

            self._set_cost_data(cost_names, cost_dims)
            self._set_constr_data(constr_names, constr_dims)


        def _set_cost_data(self, 
                        names: List[str] = None,
                        dims: List[str] = None):

            self.cost_names = names
            self.cost_dims = dims

            if (names is not None) and (self.cost_names is None) and \
                self.is_server:
                
                exception = f"Cost enabled but no cost_names list was provided"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            if (dims is not None) and (self.cost_dims is None) and \
                self.is_server:
                
                exception = f"Cost enabled but no cost_dims list was provided"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            if self.is_server and (not (len(self.cost_names) == len(self.cost_dims))):
                
                exception = f"Cost names dimension {len(self.cost_names)} " + \
                    f"does not match dim. vector length {len(self.cost_dims)}"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            if self.is_server:

                self.enable_costs = True
                self.n_costs = len(self.cost_names)

        def _set_constr_data(self, 
                        names: List[str] = None,
                        dims: List[str] = None):

            self.constr_names = names
            self.constr_dims = dims

            if (names is not None) and (self.constr_names is None) and \
                self.is_server:
                
                exception = "Constraints enabled but no cost_names list was provided"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            if (dims is not None) and (self.constr_dims is None) and \
                self.is_server:
                
                exception = "Cost enabled but no constr_dims list was provided"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            if self.is_server and (not (len(self.constr_names) == len(self.constr_dims))):
                
                exception = f"Cost names dimension {len(self.constr_names)} " + \
                    f"does not match dim. vector length {len(self.constr_dims)}"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            if self.is_server:

                self.enable_constr = True
                self.n_constr = len(self.constr_names)

    def __init__(self,
            config: Config = None,
            namespace = "",
            rhc_index = 0,
            is_server = False, 
            n_nodes: int = -1, 
            n_contacts: int = -1,
            n_jnts: int = -1,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        self.rhc_index = rhc_index

        # appending controller index to namespace
        self.namespace = namespace + "_n_" + str(self.rhc_index)

        if config is not None:

            self.config = config
        
        else:
            
            # use defaults
            self.config = self.Config()

        self.finalized = False

        self.q = None
        self.v = None
        self.a = None
        self.a_dot = None
        self.f = None
        self.f_dot = None
        self.eff = None
        self.costs = None
        self.cnstr = None
            
        if self.config.enable_q:
            
            self.q = self.Q(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 4 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_v:

            self.v = self.V(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_a:

            self.a = self.A(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_a_dot:

            self.a_dot = self.ADot(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_f:

            self.f = self.F(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
            
        if self.config.enable_f_dot:

            self.f_dot = self.FDot(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_eff:

            self.eff = self.Eff(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
            
        if self.config.enable_costs:

            self.costs = self.RHCosts(names = self.config.cost_names, # not needed if client
                    dimensions = self.config.cost_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        if self.config.enable_constr:

            self.cnstr = self.RHConstr(names = self.config.constr_names, # not needed if client
                    dimensions = self.config.constr_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection)
        
        self._is_running = False
    
    def is_running(self):

        return self._is_running
    
    def run(self):

        if self.q is not None:

            self.q.run()
        
        if self.v is not None:

            self.v.run()
        
        if self.a is not None:

            self.a.run()
        
        if self.a_dot is not None:

            self.a_dot.run()
        
        if self.f is not None:

            self.f.run()
            
        if self.f_dot is not None:

            self.f_dot.run()
        
        if self.eff is not None:

            self.eff.run()
            
        if self.costs is not None:

            self.costs.run()
        
        if self.cnstr is not None:

            self.cnstr.run()

        self.finalized = True

        self._is_running = True

    def synch(self):
        
        # to be used to read updated data 
        # (before calling any read method)
        # it synchs all available data
        
        if self.q is not None:

            self.q.synch()
        
        if self.v is not None:

            self.v.synch()
        
        if self.a is not None:

            self.a.synch()
        
        if self.a_dot is not None:

            self.a_dot.synch()
        
        if self.f is not None:

            self.f.synch()
            
        if self.f_dot is not None:

            self.f_dot.synch()
        
        if self.eff is not None:

            self.eff.synch()
            
        if self.costs is not None:

            self.costs.synch()
        
        if self.cnstr is not None:

            self.cnstr.synch()

    def close(self):

        if self.q is not None:

            self.q.close()
        
        if self.v is not None:

            self.v.close()
        
        if self.a is not None:

            self.a.close()
        
        if self.a_dot is not None:

            self.a_dot.close()
        
        if self.f is not None:

            self.f.close()
            
        if self.f_dot is not None:

            self.f_dot.close()
        
        if self.eff is not None:

            self.eff.close()
            
        if self.costs is not None:

            self.costs.close()
        
        if self.cnstr is not None:

            self.cnstr.close()

    def _check_fin_or_throw(self,
                        name: str):

        if not self.finalized:
        
            exception = "RhcInternal not initialized. Did you call the run()?"

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
    def write_q(self, 
                data: np.ndarray = None,
                wait = True):
        
        self._check_fin_or_throw("write_q")
                    
        if (self.q is not None) and (data is not None):
            
            if wait:
                
                self.q.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.q.write(data=data,
                        row_index=0, col_index=0)
    
    def write_v(self, 
            data: np.ndarray = None,
            wait = True):
        
        self._check_fin_or_throw("write_v")
        
        if (self.v is not None) and (data is not None):
            
            if wait:
                
                self.v.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.v.write(data=data,
                        row_index=0, col_index=0)

    def write_a(self, 
            data: np.ndarray = None,
            wait = True):
        
        self._check_fin_or_throw("write_a")
        
        if (self.a is not None) and (data is not None):
            
            if wait:
                
                self.a.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.a.write(data=data,
                        row_index=0, col_index=0)
            
    def write_a_dot(self, 
        data: np.ndarray = None,
        wait = True):

        self._check_fin_or_throw("write_a_dot")
        
        if (self.a_dot is not None) and (data is not None):
            
            if wait:
                
                self.a_dot.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.a_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f(self, 
        data: np.ndarray = None,
        wait = True):
        
        self._check_fin_or_throw("write_f")
        
        if (self.f is not None) and (data is not None):
            
            if wait:
                
                self.f.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.f.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f_dot(self, 
        data: np.ndarray = None,
        wait = True):

        self._check_fin_or_throw("write_f_dot")
        
        if (self.f is not None) and (data is not None):
            
            if wait:
                
                self.f_dot.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.f_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_eff(self, 
        data: np.ndarray = None,
        wait = True):

        self._check_fin_or_throw("write_eff")

        if (self.eff is not None) and (data is not None):
            
            if wait:
                
                self.eff.write_wait(data=data,
                        row_index=0, col_index=0)
            else:

                self.eff.write(data=data,
                        row_index=0, col_index=0)
                
    def write_cost(self, 
                cost_name: str,
                data: np.ndarray = None,
                wait = True):

        self._check_fin_or_throw("write_cost")
        
        if (self.costs is not None) and (data is not None):

            self.costs.write(data = data, 
                            name=cost_name,
                            wait=wait)
    
    def read_cost(self, 
            cost_name: str,
            wait = True):
        
        self._check_fin_or_throw("read_cost")

        if self.costs is not None:
            
            return self.costs.get(cost_name)
        
        else:
            
            exception = "Cannot retrieve costs. Make sure to provide cost names and dims to Config."

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
                    
    def write_constr(self, 
                constr_name: str,
                data: np.ndarray = None,
                wait = True):

        self._check_fin_or_throw("write_constr")
        
        if (self.cnstr is not None) and (data is not None):
            
            self.cnstr.write(data = data, 
                            name=constr_name,
                            wait=wait)
            
    def read_constr(self, 
            constr_name,
            wait = True):
        
        self._check_fin_or_throw("read_constr")
        
        if self.cnstr is not None:
            
            return self.cnstr.get(constr_name)
        
        else:
            
            exception = "Cannot retrieve constraints. Make sure to provide cost names and dims to Config."

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
class RhcRefs(SharedDataBase):

    def __init__(self):

        self._is_running = False
    
    def run(self):

        a = 1
    
    def close(self):

        a = 2