from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype 
from EigenIPC.PyEigenIPC import Journal

from mpc_hive.utilities.shared_data.abstractions import SharedDataBase
import numpy as np

from typing import List


def _flatten_shared_mem(shared_mem):

    if shared_mem is None:
        return []

    if isinstance(shared_mem, (list, tuple)):
        flattened = []
        for item in shared_mem:
            flattened.extend(_flatten_shared_mem(item))
        return flattened

    return [shared_mem]

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
            with_torch_view: bool = False,
            optimize_mem: bool = False):
        
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
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)
        
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
                if self._with_gpu_mirror:
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
    
    def get_shared_mem(self):

        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(super().get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.shared_jnt_names.get_shared_mem()))

        return shared_mems

    def close(self):
        super().close()
        if self.shared_jnt_names is not None:
            self.shared_jnt_names.close()

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
            fill_value = 0,
            optimize_mem: bool = False):
        
        basename = "RootState" 
        
        n_cols = 22 # p, q, v, omega, lin. acc, ang. acc., normalized gravity

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
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)
        
        if q_remapping is not None:
            self.set_q_remapping(q_remapping)
            
        # views of the underlying memory view of the 
        # actual shared memory (crazy, eh?)

        # cpu
        self._p = None
        self._q = None
        self._v = None
        self._omega = None
        self._a = None
        self._alpha = None

        self._q_full = None # full root configuration (pos + quaternion)
        self._twist = None # full root velocity (lin. + angular)
        self._a_full = None

        self._gn = None

        # gpu 
        self._p_gpu = None
        self._q_gpu = None
        self._v_gpu = None
        self._omega_gpu = None
        self._a_gpu = None
        self._alpha_gpu = None

        self._q_full_gpu = None
        self._twist_gpu = None 
        self._a_full_gpu = None

        self._gn_gpu = None
        
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

            self._a = self.get_torch_mirror()[:, 13:16].view(self.n_robots, 3)
            self._alpha = self.get_torch_mirror()[:, 16:19].view(self.n_robots, 3)
            self._a_full = self.get_torch_mirror()[:, 13:19].view(self.n_robots, 6)

            self._gn = self.get_torch_mirror()[:, 19:22].view(self.n_robots, 3)
        else:
            self._p = self.get_numpy_mirror()[:, 0:3].view()
            self._q = self.get_numpy_mirror()[:, 3:7].view()
            self._q_full = self.get_numpy_mirror()[:, 0:7].view()

            self._v = self.get_numpy_mirror()[:, 7:10].view()
            self._omega = self.get_numpy_mirror()[:, 10:13].view()
            self._twist = self.get_numpy_mirror()[:, 7:13].view()

            self._a = self.get_numpy_mirror()[:, 13:16].view()
            self._alpha = self.get_numpy_mirror()[:, 16:19].view()
            self._a_full = self.get_numpy_mirror()[:, 13:19].view()

            self._gn = self.get_numpy_mirror()[:, 19:22].view()

        if self.gpu_mirror_exists():

            # gpu views
            self._p_gpu = self._gpu_mirror[:, 0:3].view(self.n_robots, 3)
            self._q_gpu = self._gpu_mirror[:, 3:7].view(self.n_robots, 4)
            self._q_full_gpu = self._gpu_mirror[:, 0:7].view(self.n_robots, 7)

            self._v_gpu = self._gpu_mirror[:, 7:10].view(self.n_robots, 3)
            self._omega_gpu = self._gpu_mirror[:, 10:13].view(self.n_robots, 3)
            self._twist_gpu = self._gpu_mirror[:, 7:13].view(self.n_robots, 6)

            self._a_gpu = self._gpu_mirror[:, 13:16].view(self.n_robots, 3)
            self._alpha_gpu = self._gpu_mirror[:, 16:19].view(self.n_robots, 3)
            self._a_full_gpu = self._gpu_mirror[:, 13:19].view(self.n_robots, 6)

            self._gn_gpu = self._gpu_mirror[:, 19:22].view(self.n_robots, 3)
    
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
            elif name == "a":
                return self._a, None
            elif name == "alpha":
                return self._alpha, None
            elif name == "a_full":
                return self._a_full, None
            elif name == "gn":
                return self._gn, None
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
            elif name == "a":
                return self._a_gpu, None
            elif name == "alpha":
                return self._alpha_gpu, None
            elif name == "a_full":
                return self._a_full_gpu, None
            elif name == "gn":
                return self._gn_gpu, None
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
            fill_value = 0,
            optimize_mem: bool = False):
        
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
        
        n_cols=None # read from server if this is client
        if is_server:
            n_cols = self.n_contacts * 6 # cart. force + torques

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)

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
                    exception = f"Contact names list length {len(self.contact_names)} " + \
                        f"does not match the number of contacts {self.n_contacts}"
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
        data_length=int(internal_data.shape[1]/self.n_contacts)
        contact_idx=None
        
        if not contact_name in self.contact_names:
            contact_list = "\t".join(self.contact_names)
            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"
            Journal.log(self.__class__.__name__,
                "set_f_contact",
                exception,
                LogType.WARN,
                throw_when_excep = True)
        else:
            contact_idx = self.contact_names.index(contact_name)
        
        if robot_idxs is None:
            if contact_idx is None:
                internal_data[:, :] = data
            else:
                internal_data[:, (contact_idx*data_length):((contact_idx+1)*data_length)] = data
        else:
            if contact_idx is None:
                internal_data[robot_idxs, :] = data
            else:
                internal_data[robot_idxs, (contact_idx*data_length):((contact_idx+1)*data_length)] = data
        
    def get(self,
            data_type: str,
            contact_name: str = None,
            robot_idxs = None,
            gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        data_length=int(internal_data.shape[1]/self.n_contacts)
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
                return internal_data[:, (contact_idx*data_length):((contact_idx+1)*data_length)]
            else:
                return internal_data[robot_idxs, (contact_idx*data_length):((contact_idx+1)*data_length)]
        else:
            if robot_idxs is None:
                return internal_data[:, :]
            else:
                return internal_data[robot_idxs, :]

    def get_shared_mem(self):

        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(super().get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.shared_contact_names.get_shared_mem()))

        return shared_mems
    
    def close(self):
        super().close()
        if self.shared_contact_names is not None:
            self.shared_contact_names.close()
class HeightSensor(SharedTWrapper):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            grid_size: int = None,
            resolution: float = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            fill_value: float = 0.0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            optimize_mem: bool = False):

        basename = "HeightSensor" 

        n_cols = None
        if grid_size is not None:
            n_cols = grid_size * grid_size

        self.grid_size = grid_size
        self.n_robots = n_robots
        self.resolution = resolution

        # shared shape (grid size)
        # store [grid_size, resolution] as floats to avoid dtype juggling
        self._shape_shared = SharedTWrapper(namespace=namespace,
                        basename=basename + "Shape",
                        is_server=is_server,
                        n_rows=1,
                        n_cols=2,
                        dtype=eigenipc_dtype.Float,
                        verbose=verbose,
                        vlevel=vlevel,
                        fill_value=0,
                        safe=safe,
                        force_reconnection=force_reconnection,
                        with_gpu_mirror=False,
                        with_torch_view=False,
                        optimize_mem=False)

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)

        self._h_gpu = None
        self._h = None

    def run(self):
        super().run()
        self._shape_shared.run()

        if self.is_server:
            if self.grid_size is None or self.resolution is None:
                raise Exception("HeightSensor grid_size and resolution must be provided on server.")
            shape_view = self._shape_shared.get_numpy_mirror()
            shape_view[0, 0] = float(self.grid_size)
            shape_view[0, 1] = float(self.resolution)
            self._shape_shared.synch_all(read=False, retry=True)
        else:
            self._shape_shared.synch_all(read=True, retry=True)
            shape_view = self._shape_shared.get_numpy_mirror()
            self.grid_size = int(shape_view[0, 0])
            # resolution can be None on client init; fill from shared
            if getattr(self, "resolution", None) is None:
                self.resolution = float(shape_view[0, 1])
            self.n_cols = self.grid_size * self.grid_size
            if self.n_robots is not None:
                self.n_rows = self.n_robots
            else:
                self.n_robots = self.n_rows

        self._init_views()
    
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

        if self._with_torch_view:
            self._h = self.get_torch_mirror().view(self.n_robots, self.n_cols)
        else:
            self._h = self.get_numpy_mirror().view()

        if self.gpu_mirror_exists():
            self._h_gpu = self._gpu_mirror.view(self.n_robots, self.n_cols)

    def get(self, robot_idxs = None, gpu: bool = False):
        internal = self._h_gpu if (gpu and self._h_gpu is not None) else self._h
        if robot_idxs is None:
            return internal
        else:
            return internal[robot_idxs, :]

    def grid_shape(self):
        return (self.grid_size, self.grid_size)

    def set(self,
            data,
            data_type: str = None,
            robot_idxs = None,
            gpu: bool = False):

        internal = self._h_gpu if (gpu and self._h_gpu is not None) else self._h
        if robot_idxs is None:
            internal[:, :] = data
        else:
            internal[robot_idxs, :] = data

    def close(self):
        super().close()
        if self._shape_shared is not None:
            self._shape_shared.close()

    def get_shared_mem(self):

        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(super().get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self._shape_shared.get_shared_mem()))

        return shared_mems
    
class ContactPos(SharedTWrapper):

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
            fill_value = 0,
            optimize_mem: bool = False):
        
        basename = "ContactPos"

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
        
        n_cols=None
        if is_server:
            n_cols = self.n_contacts * 3 # cartesian pos * n_contats

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)

        self._p=None
        self._p_x=None
        self._p_y=None
        self._p_z=None

        self._p_gpu = None
        self._p_x_gpu = None
        self._p_y_gpu = None
        self._p_z_gpu = None

    def run(self):
        
        # overriding parent 

        super().run()
        
        if not self.is_server:

            self.n_robots = self.n_rows
            self.n_contacts = int(self.n_cols/3)

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
            self._p = self.get_torch_mirror()[:, :].view(self.n_robots, self.n_cols)
            self._p_x = self.get_torch_mirror()[:, 0:self.n_contacts].view(self.n_robots, self.n_contacts)
            self._p_y = self.get_torch_mirror()[:, self.n_contacts:(2*self.n_contacts)].view(self.n_robots, self.n_contacts)
            self._p_z = self.get_torch_mirror()[:, (2*self.n_contacts):(3*self.n_contacts)].view(self.n_robots, self.n_contacts)
        else:
            self._p = self.get_numpy_mirror()[:, :].view()
            self._p_x = self.get_numpy_mirror()[:, 0:self.n_contacts].view()
            self._p_y = self.get_numpy_mirror()[:, self.n_contacts:(2*self.n_contacts)].view()
            self._p_z = self.get_numpy_mirror()[:, (2*self.n_contacts):(3*self.n_contacts)].view()

        if self.gpu_mirror_exists():
            self._p_gpu = self._gpu_mirror[:, 0:self.n_cols].view(self.n_robots, 
                    self.n_cols)
            self._p_x_gpu = self._gpu_mirror[:, 0:self.n_contacts].view(self.n_robots, self.n_contacts)
            self._p_y_gpu = self._gpu_mirror[:, self.n_contacts:(2*self.n_contacts)].view(self.n_robots, self.n_contacts)
            self._p_z_gpu = self._gpu_mirror[:, (2*self.n_contacts):(3*self.n_contacts)].view(self.n_robots, self.n_contacts)
            
    def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
        if not gpu:
            if name == "p":
                return self._p
            elif name == "p_x":
                return self._p_x
            elif name == "p_y":
                return self._p_y
            elif name == "p_z":
                return self._p_z
            else:
                return None
        else:
            if name == "p":
                return self._p_gpu
            elif name == "p_x":
                return self._p_x_gpu
            elif name == "p_y":
                return self._p_y_gpu
            elif name == "p_z":
                return self._p_z_gpu
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
        data_length=int(internal_data.shape[1]/self.n_contacts)

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
                internal_data[:, (contact_idx*data_length):(contact_idx+1)*data_length] = data
        else:
            if contact_idx is None:
                internal_data[robot_idxs, :] = data
            else:
                internal_data[robot_idxs, contact_idx*data_length:(contact_idx+1)*data_length] = data
        
    def get(self,
            data_type: str,
            contact_name: str = None,
            robot_idxs = None,
            gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        data_length=int(internal_data.shape[1]/self.n_contacts)

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
                return internal_data[:, (contact_idx*data_length):((contact_idx+1)*data_length)]
            else:
                return internal_data[robot_idxs, (contact_idx*data_length):((contact_idx+1)*data_length)]
        else:
            if robot_idxs is None:
                return internal_data[:, :]
            else:
                return internal_data[robot_idxs, :]

    def get_shared_mem(self):

        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(super().get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.shared_contact_names.get_shared_mem()))

        return shared_mems

class ContactVel(SharedTWrapper):

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
            fill_value = 0,
            optimize_mem: bool = False):
        
        basename = "ContactVel"

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
        
        n_cols=None
        if is_server:
            n_cols = self.n_contacts * 3 # cartesian pos * n_contats

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = eigenipc_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = fill_value, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror,
            with_torch_view=with_torch_view,
            optimize_mem=optimize_mem)

        self._v=None
        self._v_x=None
        self._v_y=None
        self._v_z=None

        self._v_gpu = None
        self._v_x_gpu = None
        self._v_y_gpu = None
        self._v_z_gpu = None

    def run(self):
        
        # overriding parent 

        super().run()
        
        if not self.is_server:

            self.n_robots = self.n_rows
            self.n_contacts = int(self.n_cols/3)

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
            self._v = self.get_torch_mirror()[:, :].view(self.n_robots, self.n_cols)
            self._v_x = self.get_torch_mirror()[:, 0:self.n_contacts].view(self.n_robots, self.n_contacts)
            self._v_y = self.get_torch_mirror()[:, self.n_contacts:(2*self.n_contacts)].view(self.n_robots, self.n_contacts)
            self._v_z = self.get_torch_mirror()[:, (2*self.n_contacts):(3*self.n_contacts)].view(self.n_robots, self.n_contacts)
        else:
            self._v = self.get_numpy_mirror()[:, :].view()
            self._v_x = self.get_numpy_mirror()[:, 0:self.n_contacts].view()
            self._v_y = self.get_numpy_mirror()[:, self.n_contacts:(2*self.n_contacts)].view()
            self._v_z = self.get_numpy_mirror()[:, (2*self.n_contacts):(3*self.n_contacts)].view()

        if self.gpu_mirror_exists():
            self._v_gpu = self._gpu_mirror[:, 0:self.n_cols].view(self.n_robots, 
                    self.n_cols)
            self._v_x_gpu = self._gpu_mirror[:, 0:self.n_contacts].view(self.n_robots, self.n_contacts)
            self._v_y_gpu = self._gpu_mirror[:, self.n_contacts:(2*self.n_contacts)].view(self.n_robots, self.n_contacts)
            self._v_z_gpu = self._gpu_mirror[:, (2*self.n_contacts):(3*self.n_contacts)].view(self.n_robots, self.n_contacts)
            
    def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
        if not gpu:
            if name == "v":
                return self._v
            elif name == "v_x":
                return self._v_x
            elif name == "v_y":
                return self._v_y
            elif name == "v_z":
                return self._v_z
            else:
                return None
        else:
            if name == "v":
                return self._v_gpu
            elif name == "v_x":
                return self._v_x_gpu
            elif name == "v_y":
                return self._v_y_gpu
            elif name == "v_z":
                return self._v_z_gpu
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
        data_length=int(internal_data.shape[1]/self.n_contacts)

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
                internal_data[:, (contact_idx*data_length):(contact_idx+1)*data_length] = data
        else:
            if contact_idx is None:
                internal_data[robot_idxs, :] = data
            else:
                internal_data[robot_idxs, contact_idx*data_length:(contact_idx+1)*data_length] = data
        
    def get(self,
            data_type: str,
            contact_name: str = None,
            robot_idxs = None,
            gpu: bool = False):

        internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
        data_length=int(internal_data.shape[1]/self.n_contacts)

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
                return internal_data[:, (contact_idx*data_length):((contact_idx+1)*data_length)]
            else:
                return internal_data[robot_idxs, (contact_idx*data_length):((contact_idx+1)*data_length)]
        else:
            if robot_idxs is None:
                return internal_data[:, :]
            else:
                return internal_data[robot_idxs, :]
    
    def get_shared_mem(self):

        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(super().get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.shared_contact_names.get_shared_mem()))

        return shared_mems
    
class FullRobState(SharedDataBase):

    _OPT_ADD_ROOT_WRENCH = 0
    _OPT_ENABLE_HEIGHT_SENSOR = 1
    _OPT_HEIGHT_GRID_SIZE = 2
    _OPT_HEIGHT_GRID_RESOLUTION = 3

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
            enable_height_sensor: bool = False,
            height_grid_size: int = None,
            height_grid_resolution: float = None,
            with_gpu_mirror: bool = False,
            with_torch_view: bool = False,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            fill_value = 0,
            optimize_mem: bool = False,
            add_root_wrench: bool = True):

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

        self._add_root_wrench = add_root_wrench

        self._jnts_remapping = None
        self._q_remapping = q_remapping
        self._enable_height_sensor = enable_height_sensor
        self._height_grid_size = height_grid_size
        self._height_grid_resolution = height_grid_resolution

        self._safe = safe
        self._force_reconnection = force_reconnection

        self._with_gpu_mirror = with_gpu_mirror
        self._with_torch_view = with_torch_view

        self._optional_features_shared = SharedTWrapper(
            namespace=self._namespace + self._basename,
            basename="OptionalFeatures",
            is_server=self._is_server,
            n_rows=1,
            n_cols=4,
            dtype=eigenipc_dtype.Float,
            verbose=self._verbose,
            vlevel=self._vlevel,
            fill_value=0,
            safe=self._safe,
            force_reconnection=self._force_reconnection,
            with_gpu_mirror=False,
            with_torch_view=False,
            optimize_mem=False,
        )

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
                            fill_value=fill_value,
                            optimize_mem=optimize_mem)

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
                            fill_value=fill_value,
                            optimize_mem=optimize_mem)

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
                            fill_value=fill_value,
                            optimize_mem=optimize_mem)

        self.contact_wrenches_root = None
        self.height_sensor = None

        if self._is_server:
            self._init_optional_components(fill_value=fill_value, optimize_mem=optimize_mem)

        self.contact_pos = ContactPos(namespace=self._namespace + self._basename,
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
                            fill_value=fill_value,
                            optimize_mem=optimize_mem)

        self.contact_vel = ContactVel(namespace=self._namespace + self._basename,
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
                            fill_value=fill_value,
                            optimize_mem=optimize_mem)

        self._is_running = False

    def _init_optional_components(self,
            fill_value = 0,
            optimize_mem: bool = False):

        if self._add_root_wrench and self.contact_wrenches_root is None:
            self.contact_wrenches_root = ContactWrenches(namespace=self._namespace + self._basename + "Root",
                                is_server=self._is_server,
                                n_robots=self._n_robots,
                                n_contacts=1,
                                contact_names=["root"],
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                safe=self._safe,
                                force_reconnection=self._force_reconnection,
                                with_gpu_mirror=self._with_gpu_mirror,
                                with_torch_view=self._with_torch_view,
                                fill_value=fill_value,
                                optimize_mem=optimize_mem)

        if self._enable_height_sensor and self.height_sensor is None:
            if self._is_server and self._height_grid_size is None:
                Journal.log(self.__class__.__name__,
                    "_init_optional_components",
                    "Height sensor enabled but height_grid_size is None on server.",
                    LogType.EXCEP,
                    throw_when_excep=True)

            self.height_sensor = HeightSensor(namespace=self._namespace + self._basename,
                                is_server=self._is_server,
                                n_robots=self._n_robots,
                                grid_size=self._height_grid_size,
                                resolution=self._height_grid_resolution,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                safe=self._safe,
                                force_reconnection=self._force_reconnection,
                                with_gpu_mirror=self._with_gpu_mirror,
                                with_torch_view=self._with_torch_view,
                                fill_value=fill_value,
                                optimize_mem=optimize_mem)

    def _write_optional_features(self):

        features = self._optional_features_shared.get_numpy_mirror()
        features[0, self._OPT_ADD_ROOT_WRENCH] = 1.0 if self.contact_wrenches_root is not None else 0.0
        features[0, self._OPT_ENABLE_HEIGHT_SENSOR] = 1.0 if self.height_sensor is not None else 0.0
        features[0, self._OPT_HEIGHT_GRID_SIZE] = float(self._height_grid_size) if self._height_grid_size is not None else 0.0
        features[0, self._OPT_HEIGHT_GRID_RESOLUTION] = float(self._height_grid_resolution) if self._height_grid_resolution is not None else 0.0

        self._optional_features_shared.synch_all(read=False, retry=True)

    def _read_optional_features(self):

        self._optional_features_shared.synch_all(read=True, retry=True)
        features = self._optional_features_shared.get_numpy_mirror()

        self._add_root_wrench = bool(round(float(features[0, self._OPT_ADD_ROOT_WRENCH])))
        self._enable_height_sensor = bool(round(float(features[0, self._OPT_ENABLE_HEIGHT_SENSOR])))

        if self._enable_height_sensor:
            self._height_grid_size = int(round(float(features[0, self._OPT_HEIGHT_GRID_SIZE])))
            self._height_grid_resolution = float(features[0, self._OPT_HEIGHT_GRID_RESOLUTION])
        else:
            self._height_grid_size = None
            self._height_grid_resolution = None

    def __del__(self):

        self.close()

    def get_shared_mem(self):
        shared_mems = []
        shared_mems.extend(_flatten_shared_mem(self.root_state.get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.jnts_state.get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.contact_wrenches.get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.contact_pos.get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self.contact_vel.get_shared_mem()))
        shared_mems.extend(_flatten_shared_mem(self._optional_features_shared.get_shared_mem()))
        if self.contact_wrenches_root is not None:
            shared_mems.extend(_flatten_shared_mem(self.contact_wrenches_root.get_shared_mem()))
        if self.height_sensor is not None:
            shared_mems.extend(_flatten_shared_mem(self.height_sensor.get_shared_mem()))
        return shared_mems

    def n_robots(self):
        return self.root_state.getNRows()

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

        self._optional_features_shared.run()

        if self._is_server:
            self._write_optional_features()
        else:
            self._read_optional_features()
            self._init_optional_components()

        if self.contact_wrenches_root is not None:
            self.contact_wrenches_root.run()
        if self.height_sensor is not None:
            self.height_sensor.run()
        self.contact_pos.run()
        self.contact_vel.run()

        if not self._is_server:

            self._n_robots = self.jnts_state.n_robots

            self._n_jnts = self.jnts_state.n_jnts

            self._n_contacts = self.contact_wrenches.n_contacts

            self._jnt_names = self.jnts_state.jnt_names

            self._contact_names = self.contact_wrenches.contact_names
            if self.height_sensor is not None:
                self._height_grid_size = self.height_sensor.grid_size
                self._height_grid_resolution = getattr(self.height_sensor, "resolution", self._height_grid_resolution)

        self.set_jnts_remapping(jnts_remapping)

        self._is_running = True

    def synch_mirror(self,
                from_gpu: bool,
                non_blocking: bool = False):

        if self._with_gpu_mirror:
            if from_gpu:
                # synchs root_state and jnt_state (which will normally live on GPU)
                # with the shared state data using the aggregate view (normally on CPU)
                # this requires (not so nice) COPIES FROM GPU TO CPU
                self.root_state.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                self.jnts_state.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                self.contact_wrenches.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                if self.contact_wrenches_root is not None:
                    self.contact_wrenches_root.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                self.contact_pos.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                self.contact_vel.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                if self.height_sensor is not None:
                    self.height_sensor.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                self.synch_to_shared_mem()
            else:
                self.synch_from_shared_mem()
                # copy from CPU to GPU
                self.root_state.synch_mirror(from_gpu=False,non_blocking=non_blocking)
                self.jnts_state.synch_mirror(from_gpu=False,non_blocking=non_blocking)
                self.contact_wrenches.synch_mirror(from_gpu=False,non_blocking=non_blocking)
                if self.contact_wrenches_root is not None:
                    self.contact_wrenches_root.synch_mirror(from_gpu=False,non_blocking=non_blocking)

                self.contact_pos.synch_mirror(from_gpu=False,non_blocking=non_blocking)
                self.contact_vel.synch_mirror(from_gpu=True,non_blocking=non_blocking)
                if self.height_sensor is not None:
                    self.height_sensor.synch_mirror(from_gpu=False,non_blocking=non_blocking)
            #torch.cuda.synchronize() # this way we ensure that after this the state on GPU
            # is fully updated

    def synch_from_shared_mem(self, robot_idx: int = 0, robot_idx_view: int = 0):

        # reads from shared mem
        self.root_state.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.jnts_state.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_wrenches.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        if self.contact_wrenches_root is not None:
            self.contact_wrenches_root.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_pos.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_vel.synch_all(read = True, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        if self.height_sensor is not None:
            self.height_sensor.synch_all(read=True, retry=True, row_index=robot_idx, row_index_view=robot_idx_view)

    def synch_to_shared_mem(self, robot_idx: int = 0, robot_idx_view: int = 0):

        # write to shared mem
        self.root_state.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.jnts_state.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_wrenches.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        if self.contact_wrenches_root is not None:
            self.contact_wrenches_root.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_pos.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        self.contact_vel.synch_all(read = False, retry = True, row_index=robot_idx, row_index_view=robot_idx_view)
        if self.height_sensor is not None:
            self.height_sensor.synch_all(read=False, retry=True, row_index=robot_idx, row_index_view=robot_idx_view)

    def close(self):

        self.root_state.close()
        self.jnts_state.close()
        self.contact_wrenches.close()
        if self.contact_wrenches_root is not None:
            self.contact_wrenches_root.close()
            self.contact_wrenches_root = None
        self.contact_pos.close()
        self.contact_vel.close()
        if self.height_sensor is not None:
            self.height_sensor.close()
            self.height_sensor = None
        if self._optional_features_shared is not None:
            self._optional_features_shared.close()
