from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype 
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase
import numpy as np

from typing import List

# robot data abstractions describing a robot state
# (for both robot state and rhc cmds)

class JntsState(SharedTWrapper):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            n_jnts: int = None,
            jnt_names: List[str] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            fill_value: float = 0.0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False):
        
        basename = "JntsState" 

        n_cols = None
        if n_jnts is not None:
            n_cols = 4 * n_jnts # jnts config., vel., acc., torques

        self.n_jnts = n_jnts
        self.n_robots = n_robots
        self.jnt_names = jnt_names

        self._jnts_remapping = None
        self._jnts_remapping_gpu = None

        if is_server:
            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel,
                                        safe = safe,
                                        force_reconnection = force_reconnection)
        else:
            self.shared_jnt_names = StringTensorClient(
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel,
                                        safe = safe)
            
        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = sharsor_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view)
        
        # jnts
        self._q = None
        self._v = None
        self._a = None
        self._eff = None

        self._q_gpu = None
        self._v_gpu = None
        self._a_gpu = None
        self._eff_gpu = None
    
    def run(self,
        jnts_remapping: List[int] = None):
        
        # overriding parent 
        super().run()
        
        if not self.is_server:
            self.n_robots = self.n_rows
            self.n_jnts = int(self.n_cols / 4)

        self._init_views()
        
        # retrieving joint names
        self.shared_jnt_names.run()

        if self.is_server:
            if self.jnt_names is None:
                self.jnt_names = [""] * self.n_jnts
            else:
                if not len(self.jnt_names) == self.n_jnts:
                    exception = f"Joint names list length {len(self.jnt_names)} " + \
                        f"does not match the number of joints {self.n_jnts}"
                    Journal.log(self.__class__.__name__,
                        "run",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            jnt_names_written = self.shared_jnt_names.write_vec(self.jnt_names, 0)
            if not jnt_names_written:
                exception = "Could not write joint names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
        else:
            self.jnt_names = [""] * self.n_jnts
            while not self.shared_jnt_names.read_vec(self.jnt_names, 0):
                Journal.log(self.__class__.__name__,
                        "run",
                        "Could not read joint names on shared memory. Retrying...",
                        LogType.WARN,
                        throw_when_excep = True)
        self.set_jnts_remapping(jnts_remapping=jnts_remapping)

    def set_jnts_remapping(self, 
                jnts_remapping: List[int] = None):
        
        if jnts_remapping is not None:
            if not len(jnts_remapping) == self.n_jnts:
                warning = f"Provided jnt remapping length {len(jnts_remapping)} " + \
                    f"does not match n. joints {self.n_jnts}! Was this intentional?"
                Journal.log(self.__class__.__name__,
                    "set_jnts_remapping",
                    warning,
                    LogType.WARN,
                    throw_when_excep = True)
            if not len(jnts_remapping) <= self.n_jnts:
                warning = f"Provided jnt remapping length {len(jnts_remapping)}" + \
                    f"is higher than {self.n_jnts}. It should be <={self.n_jnts}"
                Journal.log(self.__class__.__name__,
                    "set_jnts_remapping",
                    warning,
                    LogType.WARN,
                    throw_when_excep = True)
            if self._with_torch_view:
                import torch
                self._jnts_remapping = torch.tensor(jnts_remapping, dtype=torch.int64)
                if self.with_gpu_mirror:
                    self._jnts_remapping_gpu = torch.tensor(jnts_remapping, dtype=torch.int64, device="cuda")
            else:
                self._jnts_remapping = np.array(jnts_remapping, dtype=np.int64)
        
    def _check_running(self,
                calling_method: str):

        if not self.is_running():
            exception = f"Underlying shared memory is not properly initialized." + \
                f"{calling_method}() cannot be used!."
            Journal.log(self.__class__.__name__,
                "_check_running",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

    def _init_views(self):

        self._check_running("_init_views")

        # jnts
        if self._with_torch_view:
            self._q = self.get_torch_mirror()[:, 0:self.n_jnts].view(self.n_robots, self.n_jnts)
            self._v = self.get_torch_mirror()[:, self.n_jnts:2 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._a = self.get_torch_mirror()[:, 2*self.n_jnts:3 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._eff = self.get_torch_mirror()[:, 3*self.n_jnts:4 * self.n_jnts].view(self.n_robots, self.n_jnts)
        else:
            self._q = self.get_numpy_mirror()[:, 0:self.n_jnts].view()
            self._v = self.get_numpy_mirror()[:, self.n_jnts:2 * self.n_jnts].view()
            self._a = self.get_numpy_mirror()[:, 2*self.n_jnts:3 * self.n_jnts].view()
            self._eff = self.get_numpy_mirror()[:, 3*self.n_jnts:4 * self.n_jnts].view()
        
        if self.gpu_mirror_exists():
            # gpu views 
            self._q_gpu = self._gpu_mirror[:, 0:self.n_jnts].view(self.n_robots, self.n_jnts)
            self._v_gpu = self._gpu_mirror[:, self.n_jnts:2 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._a_gpu = self._gpu_mirror[:, 2 * self.n_jnts:3 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._eff_gpu = self._gpu_mirror[:, 3 * self.n_jnts:4 * self.n_jnts].view(self.n_robots, self.n_jnts)

    def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
        if not gpu:
            if name == "q":
                return self._q
            elif name == "v":
                return self._v
            elif name == "a":
                return self._a
            elif name == "eff":
                return self._eff
            else:
                return None
        else:
            if name == "q":
                return self._q_gpu
            elif name == "v":
                return self._v_gpu
            elif name == "a":
                return self._a_gpu
            elif name == "eff":
                return self._eff_gpu
            else:
                return None
    
    def get_remapping(self):

        return self._jnts_remapping
    
    def set(self,
            data,
            data_type: str,
            robot_idxs= None,
            gpu: bool = False,
            no_remap:bool=False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        
        if self._jnts_remapping is None or no_remap:
            if robot_idxs is None:
                internal_data[:, :] = data
            else:
                internal_data[robot_idxs, :] = data
        else:
            if robot_idxs is None:
                internal_data[:, self._jnts_remapping] = data
            else:
                internal_data[robot_idxs, self._jnts_remapping] = data

    def get(self,
        data_type: str,
        robot_idxs = None,
        gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
            
        if self._jnts_remapping is None:
            if robot_idxs is None:
                return internal_data
            else:
                return internal_data[robot_idxs, :]
        else:
            if robot_idxs is None:
                return internal_data[:, self._jnts_remapping]
            else:
                return internal_data[robot_idxs, self._jnts_remapping]
         
class RootState(SharedTWrapper):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            q_remapping: List[int] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            fill_value = 0):
        
        basename = "RootState" 
        
        n_cols = 13 # p, q, v, omega + 

        self.n_robots = n_robots

        self._q_remapping = None
        self._q_full_remapping = None
        self._q_remapping_gpu = None
        self._q_full_remapping_gpu = None

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = sharsor_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view)
        
        if q_remapping is not None:
            self.set_q_remapping(q_remapping)
            
        # views of the underlying memory view of the 
        # actual shared memory (crazy, eh?)

        # root
        self._p = None
        self._q = None
        self._v = None
        self._omega = None
        self._q_full = None # full root configuration (pos + quaternion)
        self._twist = None # full root velocity (lin. + angular)

        self._p_gpu = None
        self._q_gpu = None
        self._v_gpu = None
        self._omega_gpu = None
        self._q_full_gpu = None
        self._twist_gpu = None 
        
    def run(self,
            q_remapping: List[int] = None):
        
        # overriding parent 
        super().run()
        if not self.is_server:
            self.n_robots = self.n_rows
        self._init_views()
        self.set_q_remapping(q_remapping)

    def get_remapping(self):

        return self._q_remapping
    
    def set_q_remapping(self, 
                q_remapping: List[int] = None):
    
        if q_remapping is not None:
            if not len(q_remapping) == 4:
                exception = f"Provided q remapping length {len(q_remapping)}" + \
                    f"is not 4!"
                Journal.log(self.__class__.__name__,
                    "set_q_remapping",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            q_remap_full_list = [0, 1, 2] + (np.array(q_remapping)+3).tolist()

            if self._with_torch_view:
                import torch
                self._q_remapping = torch.tensor(q_remapping, dtype=torch.int64)
                self._q_full_remapping = torch.tensor(q_remap_full_list, dtype=torch.int64)
                if self._with_gpu_mirror:
                    self._q_remapping_gpu = torch.tensor(q_remapping, dtype=torch.int64, device="cuda")
                    self._q_full_remapping_gpu = torch.tensor(q_remap_full_list, dtype=torch.int64, device="cuda")
            else:
                self._q_remapping = np.array(q_remapping, dtype=np.int64)
                self._q_full_remapping = np.array(q_remap_full_list, dtype=np.int64)

    def _init_views(self):

        # root
        if self._with_torch_view:
            self._p = self.get_torch_mirror()[:, 0:3].view(self.n_robots, 3)
            self._q = self.get_torch_mirror()[:, 3:7].view(self.n_robots, 4)
            self._q_full = self.get_torch_mirror()[:, 0:7].view(self.n_robots, 7)

            self._v = self.get_torch_mirror()[:, 7:10].view(self.n_robots, 3)
            self._omega = self.get_torch_mirror()[:, 10:13].view(self.n_robots, 3)
            self._twist = self.get_torch_mirror()[:, 7:13].view(self.n_robots, 6)
        else:
            self._p = self.get_numpy_mirror()[:, 0:3].view()
            self._q = self.get_numpy_mirror()[:, 3:7].view()
            self._q_full = self.get_numpy_mirror()[:, 0:7].view()

            self._v = self.get_numpy_mirror()[:, 7:10].view()
            self._omega = self.get_numpy_mirror()[:, 10:13].view()
            self._twist = self.get_numpy_mirror()[:, 7:13].view()

        if self.gpu_mirror_exists():

            # gpu views
            self._p_gpu = self._gpu_mirror[:, 0:3].view(self.n_robots, 3)
            self._q_gpu = self._gpu_mirror[:, 3:7].view(self.n_robots, 4)
            self._q_full_gpu = self._gpu_mirror[:, 0:7].view(self.n_robots, 7)

            self._v_gpu = self._gpu_mirror[:, 7:10].view(self.n_robots, 3)
            self._omega_gpu = self._gpu_mirror[:, 10:13].view(self.n_robots, 3)
            self._twist_gpu = self._gpu_mirror[:, 7:13].view(self.n_robots, 6)
    
    def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
        if not gpu:
            if name == "p":
                return self._p, None
            elif name == "q":
                return self._q, self._q_remapping
            elif name == "q_full":
                return self._q_full, self._q_full_remapping
            elif name == "v":
                return self._v, None
            elif name == "omega":
                return self._omega, None
            elif name == "twist":
                return self._twist, None
            else:
                return None, None
        else:
            if name == "p":
                return self._p_gpu, None
            elif name == "q":
                return self._q_gpu, self._q_remapping_gpu
            elif name == "q_full":
                return self._q_full_gpu, self._q_full_remapping_gpu
            elif name == "v":
                return self._v_gpu, None
            elif name == "omega":
                return self._omega_gpu, None
            elif name == "twist":
                return self._twist_gpu, None
            else:
                return None, None
    
    def set(self,
            data,
            data_type: str,
            robot_idxs= None,
            gpu: bool = False):

        internal_data, remapping = self._retrieve_data(name=data_type,
                    gpu=gpu)
        
        if remapping is None:
            if robot_idxs is None:
                internal_data[:, :] = data
            else:
                internal_data[robot_idxs, :] = data
        else:
            if robot_idxs is None:
                internal_data[:, remapping] = data
            else:
                internal_data[robot_idxs, remapping] = data

    def get(self,
        data_type: str,
        robot_idxs = None,
        gpu: bool = False):

        internal_data, remapping = self._retrieve_data(name=data_type,
                    gpu=gpu)
            
        if remapping is None:
            if robot_idxs is None:
                return internal_data
            else:
                return internal_data[robot_idxs, :]
        else:
            if robot_idxs is None:
                return internal_data[:, remapping]
            else:
                return internal_data[robot_idxs, remapping]
            
class ContactWrenches(SharedTWrapper):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            n_contacts: int = None,
            contact_names: List[str] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            fill_value = 0):
        
        basename = "ContactWrenches"

        self.n_robots = n_robots
        self.n_contacts = n_contacts
        self.contact_names = contact_names

        if is_server:
            self.shared_contact_names = StringTensorServer(length = self.n_contacts, 
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel,
                                        safe = safe,
                                        force_reconnection = force_reconnection)
        else:
            self.shared_contact_names = StringTensorClient(
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel,
                                        safe = safe)
            
        n_cols = self.n_contacts * 6 # cart. force + torques

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = sharsor_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view)

        self._f = None
        self._t = None
        self._w = None

        self._f_gpu = None
        self._t_gpu = None
        self._w_gpu = None

    def run(self):
        
        # overriding parent 

        super().run()
        
        if not self.is_server:

            self.n_robots = self.n_rows
            self.n_contacts = int(self.n_cols / 6)

        self._init_views()

        # retrieving contact names
        self.shared_contact_names.run()

        if self.is_server:
            if self.contact_names is None:
                self.contact_names = [""] * self.n_contacts
            else:
                if not len(self.contact_names) == self.n_contacts:
                    exception = f"Joint names list length {len(self.contact_names)} " + \
                        f"does not match the number of joints {self.n_contacts}"
                    Journal.log(self.__class__.__name__,
                        "run",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            written = self.shared_contact_names.write_vec(self.contact_names, 0)
            if not written:
                exception = "Could not write contact names on shared memory!"
                Journal.log(self.__class__.__name__,
                        "run",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
        else:
            self.contact_names = [""] * self.n_contacts
            while not self.shared_contact_names.read_vec(self.contact_names, 0):
                Journal.log(self.__class__.__name__,
                    "run",
                    "Could not read contact names on shared memory. Retrying...",
                    LogType.WARN,
                    throw_when_excep = True)
            
    def _init_views(self):

        if self._with_torch_view:
            self._f = self.get_torch_mirror()[:, 0:self.n_contacts * 3].view(self.n_robots, 
                                                                    self.n_contacts * 3)
            self._t = self.get_torch_mirror()[:, (self.n_contacts * 3):(self.n_contacts * 6)].view(self.n_robots, 
                                                                    self.n_contacts * 3)
            self._w = self.get_torch_mirror()[:, :].view(self.n_robots, self.n_contacts * 6)
        else:
            self._f = self.get_numpy_mirror()[:, 0:self.n_contacts * 3].view()
            self._t = self.get_numpy_mirror()[:, (self.n_contacts * 3):(self.n_contacts * 6)].view()
            self._w = self.get_numpy_mirror()[:, :].view()

        if self.gpu_mirror_exists():
            self._f_gpu = self._gpu_mirror[:, 0:self.n_contacts * 3].view(self.n_robots, 
                                                                self.n_contacts * 3)
            self._t_gpu = self._gpu_mirror[:, self.n_contacts * 3:self.n_contacts * 6].view(self.n_robots, 
                                                                    self.n_contacts * 3)
            self._w_gpu = self._gpu_mirror[:, :].view(self.n_robots, self.n_contacts * 6)
    
    def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
        if not gpu:
            if name == "f":
                return self._f
            elif name == "t":
                return self._t
            elif name == "w":
                return self._w
            else:
                return None
        else:
            if name == "f":
                return self._f_gpu
            elif name == "t":
                return self._t_gpu
            elif name == "w":
                return self._w_gpu
            else:
                return None
     
    def set(self,
            data,
            data_type: str,
            contact_name: str,
            robot_idxs = None,
            gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        
        if not contact_name in self.contact_names:
            contact_list = "\t".join(self.contact_names)
            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"
            Journal.log(self.__class__.__name__,
                "set_f_contact",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        contact_idx = self.contact_names.index(contact_name)
        
        if robot_idxs is None:
            if contact_idx is None:
                internal_data[:, :] = data
            else:
                internal_data[:, (contact_idx * 3):((contact_idx+1) * 3)] = data
        else:
            if contact_idx is None:
                internal_data[robot_idxs, :] = data
            else:
                internal_data[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)] = data
        
    def get(self,
            data_type: str,
            contact_name: str = None,
            robot_idxs = None,
            gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        
        if contact_name is not None:
            if not contact_name in self.contact_names:
                contact_list = "\t".join(self.contact_names)
                exception = f"Contact name {contact_name} not in contact list [{contact_list}]"
                Journal.log(self.__class__.__name__,
                    "get_f_contact",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            contact_idx = self.contact_names.index(contact_name)
            if robot_idxs is None:
                return internal_data[:, (contact_idx * 3):((contact_idx+1) * 3)]
            else:
                return internal_data[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)]
        else:
            if robot_idxs is None:
                return internal_data[:, :]
            else:
                return internal_data[robot_idxs, :]
            
class FullRobState(SharedDataBase):

    def __init__(self,
            namespace: str,
            basename: str,
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

        self._namespace = namespace
        self._basename = basename

        self._is_server = is_server
        
        self._verbose = verbose
        self._vlevel = vlevel

        self._n_robots = n_robots
        self._n_jnts = n_jnts
        self._n_contacts = n_contacts
        self._jnt_names = jnt_names
        self._contact_names = contact_names
        
        self._jnts_remapping = None
        self._q_remapping = q_remapping

        self._safe = safe
        self._force_reconnection = force_reconnection
        
        self._with_gpu_mirror = with_gpu_mirror
        self._with_torch_view = with_torch_view

        self.root_state = RootState(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            q_remapping=self._q_remapping,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror,
                            with_torch_view=with_torch_view,
                            fill_value=fill_value)
    
        self.jnts_state = JntsState(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            n_jnts=self._n_jnts,
                            jnt_names=self._jnt_names,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror,
                            with_torch_view=with_torch_view,
                            fill_value=fill_value)
        
        self.contact_wrenches = ContactWrenches(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            n_contacts=self._n_contacts,
                            contact_names=self._contact_names,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror,
                            with_torch_view=with_torch_view,
                            fill_value=fill_value)
        
        self._is_running = False
    
    def __del__(self):

        self.close()
    
    def get_shared_mem(self):
        return [self.root_state.get_shared_mem(),
            self.jnts_state.get_shared_mem(),
            self.contact_wrenches.get_shared_mem()]
    
    def n_robots(self):

        return self._n_robots
    
    def n_jnts(self):

        return self._n_jnts
    
    def n_contacts(self):

        return self._n_contacts
    
    def jnt_names(self):

        return self._jnt_names
    
    def contact_names(self):

        return self._contact_names
    
    def is_running(self):

        return self._is_running
    
    def set_jnts_remapping(self,
                jnts_remapping: List[int] = None):

        self.jnts_state.set_jnts_remapping(jnts_remapping=jnts_remapping)
    
    def set_q_remapping(self,
                q_remapping: List[int] = None):

        self.root_state.set_q_remapping(q_remapping=q_remapping)

    def run(self,
        jnts_remapping: List[int] = None):

        self.root_state.run()

        self.jnts_state.run()

        self.contact_wrenches.run()

        if not self._is_server:

            self._n_robots = self.jnts_state.n_robots

            self._n_jnts = self.jnts_state.n_jnts

            self._n_contacts = self.contact_wrenches.n_contacts

            self._jnt_names = self.jnts_state.jnt_names

            self._contact_names = self.contact_wrenches.contact_names

        self.set_jnts_remapping(jnts_remapping)

        self._is_running = True
        
    def synch_mirror(self,
                from_gpu: bool):

        if self._with_gpu_mirror:
            if from_gpu:
                # synchs root_state and jnt_state (which will normally live on GPU)
                # with the shared state data using the aggregate view (normally on CPU)
                # this requires (not so nice) COPIES FROM GPU TO CPU
                self.root_state.synch_mirror(from_gpu=True)
                self.jnts_state.synch_mirror(from_gpu=True)
                self.contact_wrenches.synch_mirror(from_gpu=True)
                self.synch_to_shared_mem()
            else:
                self.synch_from_shared_mem()
                # copy from CPU to GPU
                self.root_state.synch_mirror(from_gpu=False)
                self.jnts_state.synch_mirror(from_gpu=False)
                self.contact_wrenches.synch_mirror(from_gpu=False)

            #torch.cuda.synchronize() # this way we ensure that after this the state on GPU
            # is fully updated
    
    def synch_from_shared_mem(self):

        # reads from shared mem
        self.root_state.synch_all(read = True, retry = True)
        self.jnts_state.synch_all(read = True, retry = True)
        self.contact_wrenches.synch_all(read = True, retry = True)

    def synch_to_shared_mem(self):

        # write to shared mem
        self.root_state.synch_all(read = False, retry = True)
        self.jnts_state.synch_all(read = False, retry = True)
        self.contact_wrenches.synch_all(read = False, retry = True)
        
    def close(self):

        self.root_state.close()
        self.jnts_state.close()
        self.contact_wrenches.close()
