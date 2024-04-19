from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase
from control_cluster_bridge.utilities.shared_data.state_encoding import FullRobState

import numpy as np

from typing import List
        
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
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            fill_value = 0):

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
            with_torch_view=with_torch_view,
            force_reconnection=force_reconnection,
            safe=safe,
            verbose=verbose,
            vlevel=vlevel,
            fill_value=fill_value)

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
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            fill_value=0):

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
            with_torch_view=with_torch_view,
            force_reconnection=force_reconnection,
            safe=safe,
            verbose=verbose,
            vlevel=vlevel,
            fill_value=fill_value)
        
class RhcRefs(SharedDataBase):
    
    class RobotFullConfigRef(FullRobState):

        def __init__(self,
                namespace: str,
                is_server: bool,
                basename: str = "",
                n_robots: int = None,
                n_jnts: int = None,
                n_contacts: int = 1,
                jnt_names: List[str] = None,
                contact_names: List[str] = None,
                q_remapping: List[int] = None,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False,
                force_reconnection: bool = False,
                safe: bool = True,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                fill_value=np.nan, # if ref is not used
                ):

            basename = basename + "RobotFullConfigRef"

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
                with_torch_view=with_torch_view,
                force_reconnection=force_reconnection,
                safe=safe,
                verbose=verbose,
                vlevel=vlevel,
                fill_value=fill_value)
    
    class Phase(SharedTWrapper):

        def __init__(self,
            namespace = "",
            basename = "",
            is_server = False, 
            n_robots: int = -1, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            with_torch_view: bool = False,
            with_gpu_mirror: bool = False,
            safe: bool = True):
        
            basename = basename + "Phase" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_robots, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Int,
                force_reconnection=force_reconnection,
                with_torch_view=with_torch_view,
                with_gpu_mirror=with_gpu_mirror,
                fill_value = -1)
            
    class ContactFlag(SharedTWrapper):

        def __init__(self,
            namespace = "",
            basename = "",
            is_server = False, 
            n_robots: int = -1, 
            n_contacts: int = -1,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            with_torch_view: bool = False,
            with_gpu_mirror: bool = False,
            safe: bool = True):
        
            basename = basename + "ContactFlag" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_robots, 
                n_cols = n_contacts, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                with_torch_view=with_torch_view,
                with_gpu_mirror=with_gpu_mirror,
                fill_value = True)

    def __init__(self,
                namespace: str,
                is_server: bool,
                n_robots: int = None,
                n_jnts: int = None,
                n_contacts: int = 1,
                jnt_names: List[str] = None,
                contact_names: List[str] = None,
                q_remapping: List[int] = None,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False,
                force_reconnection: bool = False,
                safe: bool = False,
                verbose: bool = False,
                vlevel: VLevel = VLevel.V1,
                fill_value=np.nan):
        
        self.basename = "Rhc"

        self.is_server = is_server

        self._with_gpu_mirror = with_gpu_mirror
        self._with_torch_view = with_torch_view

        self.n_robots = n_robots

        self.namespace = namespace

        self.verbose = verbose

        self.vlevel = vlevel

        self.force_reconnection = force_reconnection

        self.safe = safe

        self.rob_refs = self.RobotFullConfigRef(namespace=namespace,
                                    basename=self.basename,
                                    is_server=is_server,
                                    n_robots=n_robots,
                                    n_jnts=n_jnts,
                                    n_contacts=n_contacts,
                                    jnt_names=jnt_names,
                                    contact_names=contact_names,
                                    q_remapping=q_remapping,
                                    with_gpu_mirror=with_gpu_mirror,
                                    with_torch_view=with_torch_view,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    verbose=verbose,
                                    vlevel=vlevel,
                                    fill_value=fill_value)
        
        self.phase_id = self.Phase(namespace=namespace,
                            basename=self.basename,
                            is_server=is_server,
                            n_robots=n_robots,
                            verbose=verbose,
                            vlevel=vlevel,
                            force_reconnection=force_reconnection,
                            with_gpu_mirror=with_gpu_mirror,
                            with_torch_view=with_torch_view,
                            safe=safe)

        self.contact_flags = None

        self._is_runnning = False

    def __del__(self):

        self.close()

    def is_running(self):
    
        return self._is_runnning
    
    def get_shared_mem(self):
        return self.rob_refs.get_shared_mem() + [
            self.phase_id.get_shared_mem(),
            self.contact_flags.get_shared_mem()]
    
    def run(self):

        self.rob_refs.run()
        self.phase_id.run()

        self.n_contacts = self.rob_refs.n_contacts()
        
        self.n_robots = self.rob_refs.n_robots()    
        
        self.contact_flags = self.ContactFlag(namespace=self.namespace,
                            basename=self.basename,
                            is_server=self.is_server,
                            n_robots=self.n_robots,
                            n_contacts=self.n_contacts,
                            verbose=self.verbose,
                            vlevel=self.vlevel,
                            force_reconnection=self.force_reconnection,
                            with_gpu_mirror=self._with_gpu_mirror,
                            with_torch_view=self._with_torch_view,
                            safe=self.safe)
        self.contact_flags.run()

        self._is_runnning = True

    def close(self):
        
        if self.is_running():
            
            self.rob_refs.close()
            self.phase_id.close()
            self.contact_flags.close()

            self._is_runnning = False

class RhcStatus(SharedDataBase):
    
    class FailFlagView(SharedTWrapper):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
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
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = False)
    
    class ResetFlagView(SharedTWrapper):
        
        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
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
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = False)
    
    class TriggerFlagView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
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
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = False)
    
    class ActivationFlagView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
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
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = False)
    
    class RegistrationFlagView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
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
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = False)
            
    class ControllersCounterView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "ClusterControllersCounter" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = 1, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Int,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = 0)
    
    class FailsCounterView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "ClusterControllerFailsCounter" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Int,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = 0)
            
    class RhcCostView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "RhcCost" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Float,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = np.nan)
    
    class RhcCnstrViolationView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "RhcCnstrViolation" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Float,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = np.nan)
    
    class RhcNodesCostView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                n_nodes: int = -1,
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "RhcNodesCost" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Float,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = 0)
    
    class RhcNodesCnstrViolationView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                n_nodes: int = -1,
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "RhcNodesCnstrViolation" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Float,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = 0)
    
    class RhcNIterationsView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                cluster_size: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "RhcNIterations" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = cluster_size, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = False, # boolean operations are atomic on 64 bit systems
                dtype=dtype.Float,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=with_torch_view,
                fill_value = np.nan)
                    
    def __init__(self, 
            is_server = False, 
            cluster_size: int = -1, 
            n_nodes: int = -1,
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False):

        self.is_server = is_server

        self.cluster_size = cluster_size
        self.n_nodes = n_nodes

        self.namespace = namespace

        self.verbose = verbose

        self.vlevel = vlevel

        self.with_gpu_mirror = with_gpu_mirror

        self.fails = self.FailFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.resets = self.ResetFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.trigger = self.TriggerFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.activation_state = self.ActivationFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.registration = self.RegistrationFlagView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)

        self.controllers_counter = self.ControllersCounterView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.controllers_fail_counter = self.FailsCounterView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size,
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)

        self.rhc_cost = self.RhcCostView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)

        self.rhc_constr_viol = self.RhcCnstrViolationView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self.rhc_nodes_cost = self.RhcNodesCostView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                n_nodes=self.n_nodes,
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)

        self.rhc_nodes_constr_viol = self.RhcNodesCnstrViolationView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                n_nodes=self.n_nodes,
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)

        self.rhc_n_iter = self.RhcNIterationsView(namespace=self.namespace, 
                                is_server=self.is_server, 
                                cluster_size=self.cluster_size, 
                                verbose=self.verbose, 
                                vlevel=vlevel,
                                force_reconnection=force_reconnection,
                                with_gpu_mirror=with_gpu_mirror,
                                with_torch_view=with_torch_view)
        
        self._is_runnning = False

        self._acquired_reg_sem = False
        
    def __del__(self):

        self.close()

    def is_running(self):
    
        return self._is_runnning
    
    def get_shared_mem(self):
        return [self.fails.get_shared_mem(),
            self.resets.get_shared_mem(),
            self.trigger.get_shared_mem(),
            self.activation_state.get_shared_mem(),
            self.registration.get_shared_mem(),
            self.controllers_counter.get_shared_mem(),
            self.controllers_fail_counter.get_shared_mem(),
            self.rhc_cost.get_shared_mem(),
            self.rhc_constr_viol.get_shared_mem(),
            self.rhc_n_iter.get_shared_mem(),
            self.rhc_nodes_cost.get_shared_mem(),
            self.rhc_nodes_constr_viol.get_shared_mem()]
    
    def run(self):

        self.resets.run()
        self.trigger.run()
        self.fails.run()
        self.activation_state.run()
        self.registration.run()
        self.controllers_counter.run()
        self.controllers_fail_counter.run()
        self.rhc_cost.run()
        self.rhc_constr_viol.run()
        self.rhc_nodes_cost.run()
        self.rhc_nodes_constr_viol.run()
        self.rhc_n_iter.run()

        if not self.is_server:
    
            self.cluster_size = self.trigger.n_rows
            self.n_nodes = self.rhc_nodes_cost.n_cols

        self._is_runnning = True

    def close(self):
        
        if self.is_running():
            
            self.trigger.close()
            self.resets.close()
            self.fails.close()    
            self.activation_state.close()
            self.registration.close()
            self.controllers_counter.close()
            self.controllers_fail_counter.close()
            self.rhc_n_iter.close()
            self.rhc_cost.close()
            self.rhc_constr_viol.close()
            self.rhc_nodes_cost.close()
            self.rhc_nodes_constr_viol.close()

            self._is_runnning = False

class RhcInternal(SharedDataBase):

    # class for sharing internal data of a 
    # receding-horizon controller

    class Q(SharedTWrapper):

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
            
            basename = "Q" # configuration vector

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
    
    class V(SharedTWrapper):

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
            
            basename = "V" # velocity vector

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
    
    class A(SharedTWrapper):

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
    
    class ADot(SharedTWrapper):

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
            
            basename = "ADot" # jerk vector

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
    
    class F(SharedTWrapper):

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
            
            basename = "F" # cartesian force vector

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
            
    class FDot(SharedTWrapper):

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
            
            basename = "FDot" # yank vector

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
            
    class Eff(SharedTWrapper):

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
            
            basename = "Eff" # hardcoded

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
            
            basename = "RhcCosts"

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
            
            basename = "RhcConstraints"

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
            enable_q: bool = False, 
            enable_v: bool = False, 
            enable_a: bool = False,
            enable_a_dot: bool = False, 
            enable_f: bool = False, 
            enable_f_dot: bool = False, 
            enable_eff: bool = False, 
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
            n_nodes: int = -1, 
            n_contacts: int = -1,
            n_jnts: int = -1,
            jnt_names: List[str] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            safe: bool = True):

        self.rhc_index = rhc_index
        self._basename = "RhcInternal"

        self._verbose = verbose
        self._vlevel = vlevel

        self._jnt_names = jnt_names
        self._n_jnts = n_jnts

        # appending controller index to namespace
        self.namespace = self._basename + namespace + "_n_" + str(self.rhc_index)
        
        if config is not None:
            self.config = config
        else:
            # use defaults
            self.config = self.Config()

        self.q = None
        self.v = None
        self.a = None
        self.a_dot = None
        self.f = None
        self.f_dot = None
        self.eff = None
        self.costs = None
        self.cnstr = None
        
        self._shared_jnt_names = None

        self._is_server = config.is_server

        if self.config.enable_q:
            self.q = self.Q(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 3 + 4 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_v:
            self.v = self.V(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_a:
            self.a = self.A(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_a_dot:
            self.a_dot = self.ADot(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_f:
            self.f = self.F(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
            
        if self.config.enable_f_dot:
            self.f_dot = self.FDot(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 6 * n_contacts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_eff:
            self.eff = self.Eff(namespace = self.namespace,
                    is_server = self._is_server, 
                    n_dims = 3 + 3 + n_jnts, 
                    n_nodes = n_nodes, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
            
        if self.config.enable_costs:
            self.costs = self.RHCosts(names = self.config.cost_names, # not needed if client
                    dimensions = self.config.cost_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = self._is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self.config.enable_constr:
            self.cnstr = self.RHConstr(names = self.config.constr_names, # not needed if client
                    dimensions = self.config.constr_dims, # not needed if client
                    n_nodes = n_nodes, # not needed if client 
                    namespace = self.namespace,
                    is_server = self._is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    force_reconnection=force_reconnection,
                    safe=safe)
        
        if self._is_server:
            self._shared_jnt_names = StringTensorServer(length = len(self._jnt_names), 
                                        basename = self._basename + "Names", 
                                        name_space = self.namespace,
                                        verbose = self._verbose, 
                                        vlevel = self._vlevel,
                                        safe = safe,
                                        force_reconnection = force_reconnection)
        else:
            self._shared_jnt_names = StringTensorClient(
                                        basename = self._basename + "Names", 
                                        name_space = self.namespace,
                                        verbose = self._verbose, 
                                        vlevel = self._vlevel,
                                        safe = safe)
            
        self._is_running = False
    
    def is_running(self):

        return self._is_running
    
    def get_shared_mem(self):
        return [self.fails.get_shared_mem(),
            self.q.get_shared_mem(),
            self.v.get_shared_mem(),
            self.a.get_shared_mem(),
            self.a_dot.get_shared_mem(),
            self.f.get_shared_mem(),
            self.f_dot.get_shared_mem(),
            self.eff.get_shared_mem(),
            self.costs.get_shared_mem(),
            self.cnstr.get_shared_mem(),
            self._shared_jnt_names.get_shared_mem()]
                
    def jnt_names(self):

        return self._jnt_names
        
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

        self._shared_jnt_names.run()

        if self._is_server:
            
            if self._jnt_names is None:
                self._jnt_names = [""] * self._n_jnts
            else:
                if not len(self._jnt_names) == self._n_jnts:
                    exception = f"Joint names list length {len(self._jnt_names)} " + \
                        f"does not match the number of joints {self._n_jnts}"
                    Journal.log(self.__class__.__name__,
                        "run",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            jnt_names_written = self._shared_jnt_names.write_vec(self._jnt_names, 0)
            if not jnt_names_written:
                exception = "Could not write joint names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                    
        else:
            
            if self.q is not None:

                self._n_jnts = self.q.n_rows - 7
                self._jnt_names = [""] * self._n_jnts
                while not self._shared_jnt_names.read_vec(self._jnt_names, 0):
                    Journal.log(self.__class__.__name__,
                        "run",
                        "Could not read joint names on shared memory. Retrying...",
                        LogType.WARN,
                        throw_when_excep = True)

        self._is_running = True

    def synch(self, read = True):
        
        # to be used to read updated data 
        # (before calling any read method)
        # it synchs all available data
        
        if self.q is not None:
            self.q.synch_all(read=read, retry=True)
        
        if self.v is not None:
            self.v.synch_all(read=read, retry=True)
        
        if self.a is not None:
            self.a.synch_all(read=read, retry=True)
        
        if self.a_dot is not None:
            self.a_dot.synch_all(read=read, retry=True)
        
        if self.f is not None:
            self.f.synch_all(read=read, retry=True)
            
        if self.f_dot is not None:
            self.f_dot.synch_all(read=read, retry=True)
        
        if self.eff is not None:
            self.eff.synch_all(read=read, retry=True)
            
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
        
        if self._shared_jnt_names is not None:
            self._shared_jnt_names.close()

    def _check_running_or_throw(self,
                        name: str):

        if not self.is_running():
            exception = "RhcInternal not initialized. Did you call the run()?"
            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
    def write_q(self, 
                data: np.ndarray = None,
                retry = True):
        
        self._check_running_or_throw("write_q")
        if (self.q is not None) and (data is not None):
            
            if retry:
                self.q.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.q.write(data=data,
                        row_index=0, col_index=0)
    
    def write_v(self, 
            data: np.ndarray = None,
            retry = True):
        
        self._check_running_or_throw("write_v")
        if (self.v is not None) and (data is not None):
            if retry:
                self.v.write_retry(data=data,
                        row_index=0, col_index=0)
            else:

                self.v.write(data=data,
                        row_index=0, col_index=0)

    def write_a(self, 
            data: np.ndarray = None,
            retry = True):
        
        self._check_running_or_throw("write_a")
        if (self.a is not None) and (data is not None):    
            if retry:
                self.a.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.a.write(data=data,
                        row_index=0, col_index=0)
            
    def write_a_dot(self, 
        data: np.ndarray = None,
        retry = True):

        self._check_running_or_throw("write_a_dot")
        if (self.a_dot is not None) and (data is not None):
            if retry:
                self.a_dot.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.a_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f(self, 
        data: np.ndarray = None,
        retry = True):
        
        self._check_running_or_throw("write_f")  
        if (self.f is not None) and (data is not None): 
            if retry:
                self.f.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.f.write(data=data,
                        row_index=0, col_index=0)
    
    def write_f_dot(self, 
        data: np.ndarray = None,
        retry = True):

        self._check_running_or_throw("write_f_dot")
        if (self.f is not None) and (data is not None):
            if retry:
                self.f_dot.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.f_dot.write(data=data,
                        row_index=0, col_index=0)
    
    def write_eff(self, 
        data: np.ndarray = None,
        retry = True):

        self._check_running_or_throw("write_eff")
        if (self.eff is not None) and (data is not None):
            if retry:
                self.eff.write_retry(data=data,
                        row_index=0, col_index=0)
            else:
                self.eff.write(data=data,
                        row_index=0, col_index=0)
                
    def write_cost(self, 
                cost_name: str,
                data: np.ndarray = None,
                retry = True):

        self._check_running_or_throw("write_cost")
        if (self.costs is not None) and (data is not None):
            self.costs.write(data = data, 
                            name=cost_name,
                            retry=retry)
    
    def read_cost(self, 
            cost_name: str,
            retry = True):
        
        self._check_running_or_throw("read_cost")
        if self.costs is not None:
            return self.costs.get(cost_name)
        else:
            exception = "Cannot retrieve costs. Make sure to provide cost names and dims to Config."
            Journal.log(self.__class__.__name__,
                "read_cost",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
                    
    def write_constr(self, 
                constr_name: str,
                data: np.ndarray = None,
                retry = True):
        
        self._check_running_or_throw("write_constr")
        if (self.cnstr is not None) and (data is not None):
            self.cnstr.write(data = data, 
                            name=constr_name,
                            retry=retry)
            
    def read_constr(self, 
            constr_name,
            retry = True):
        
        self._check_running_or_throw("read_constr")
        if self.cnstr is not None:
            return self.cnstr.get(constr_name)
        else:
            exception = "Cannot retrieve constraints. Make sure to provide cost names and dims to Config."
            Journal.log(self.__class__.__name__,
                "read_constr",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)