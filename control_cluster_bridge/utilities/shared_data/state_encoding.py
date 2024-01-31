from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype 
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase
import numpy as np

import torch

from typing import List

# robot data abstractions describing a robot state
# (for both robot state and rhc cmds)

class JntsState(SharedDataView):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            n_jnts: int = None,
            jnt_names: List[str] = None,
            jnts_remapping: List[int] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            fill_value: float = 0.0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = True):
        
        basename = "JntsState" 

        n_cols = None

        if n_jnts is not None:

            n_cols = 4 * n_jnts # jnts config., vel., acc., torques

        self.n_jnts = n_jnts
        self.n_robots = n_robots
        self.jnt_names = jnt_names

        self._jnts_remapping = None
        if jnts_remapping is not None:
            self._jnts_remapping = torch.tensor(jnts_remapping)

        if is_server:

            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)

        else:

            self.shared_jnt_names = StringTensorClient(
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
            
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
            with_gpu_mirror=with_gpu_mirror)

        # jnts
        self._q = None
        self._v = None
        self._a = None
        self._eff = None

        self._q_gpu = None
        self._v_gpu = None
        self._a_gpu = None
        self._eff_gpu = None
    
    def run(self):
        
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
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                    
        else:
            
            self.jnt_names = [""] * self.n_jnts

            jnt_names_read = self.shared_jnt_names.read_vec(self.jnt_names, 0)

            if not jnt_names_read:
                
                exception = "Could not read joint names on shared memory!"

                Journal.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
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
        self._q = self.torch_view[:, 0:self.n_jnts].view(self.n_robots, self.n_jnts)
        self._v = self.torch_view[:, self.n_jnts:2 * self.n_jnts].view(self.n_robots, self.n_jnts)
        self._a = self.torch_view[:, 2*self.n_jnts:3 * self.n_jnts].view(self.n_robots, self.n_jnts)
        self._eff = self.torch_view[:, 3*self.n_jnts:4 * self.n_jnts].view(self.n_robots, self.n_jnts)

        if self.gpu_mirror_exists():

            # gpu views
            self._q_gpu = self._gpu_mirror[:, 0:self.n_jnts].view(self.n_robots, self.n_jnts)
            self._v_gpu = self._gpu_mirror[:, self.n_jnts:2 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._a_gpu = self._gpu_mirror[:, 2 * self.n_jnts:3 * self.n_jnts].view(self.n_robots, self.n_jnts)
            self._eff_gpu = self._gpu_mirror[:, 3 * self.n_jnts:4 * self.n_jnts].view(self.n_robots, self.n_jnts)

    def _check_mirror_of_throw(self,
                        name: str):

        if not self.gpu_mirror_exists():

            exception = f"GPU mirror is not available!"

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

    def get_q(self,
            robot_idx: int = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._q[:, :]
                
                else:

                    return self._q[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._q_gpu[:, :]
                
                else:

                    return self._q_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._q[:, self._jnts_remapping]
                
                else:

                    return self._q[robot_idx, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._q_gpu[:, self._jnts_remapping]
                
                else:

                    return self._q_gpu[robot_idx, self._jnts_remapping].view(1, -1)

    def get_v(self,
            robot_idx: int = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._v[:, :]
                
                else:

                    return self._v[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idx is None:

                    return self._v_gpu[:, :]
                
                else:

                    return self._v_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._v[:, self._jnts_remapping]
                
                else:

                    return self._v[robot_idx, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idx is None:

                    return self._v_gpu[:, self._jnts_remapping]
                
                else:

                    return self._v_gpu[robot_idx, self._jnts_remapping].view(1, -1)

    def get_a(self,
            robot_idx: int = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._a[:, :]
                
                else:

                    return self._a[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idx is None:

                    return self._a_gpu[:, :]
                
                else:

                    return self._a_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._a[:, self._jnts_remapping]
                
                else:

                    return self._a[robot_idx, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idx is None:

                    return self._a_gpu[:, self._jnts_remapping]
                
                else:

                    return self._a_gpu[robot_idx, self._jnts_remapping].view(1, -1)

    def get_eff(self,
        robot_idx: int = None,
        gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._eff[:, :]
                
                else:

                    return self._eff[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_eff")

                if robot_idx is None:

                    return self._eff_gpu[:, :]
                
                else:

                    return self._eff_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._eff[:, self._jnts_remapping]
                
                else:

                    return self._eff[robot_idx, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_eff")

                if robot_idx is None:

                    return self._eff_gpu[:, self._jnts_remapping]
                
                else:

                    return self._eff_gpu[robot_idx, self._jnts_remapping].view(1, -1)
         
class RootState(SharedDataView):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            q_remapping: List[int] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = True):
        
        basename = "RootState" 

        n_cols = 13 # p, q, v, omega + 

        self.n_robots = n_robots

        self._q_remapping = None
        if q_remapping is not None:
            self._q_remapping = torch.tensor(q_remapping)

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = sharsor_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = 0, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror)

        # views of the underlying memory view of the 
        # actual shared memory (crazy, eh?)

        # root
        self._p = None
        self._q = None
        self._v = None
        self._omega = None

        self._p_gpu = None
        self._q_gpu = None
        self._v_gpu = None
        self._omega_gpu = None
        
    def run(self):
        
        # overriding parent 

        super().run()
        
        if not self.is_server:

            self.n_robots = self.n_rows

        self._init_views()

    def _init_views(self):

        # root
        self._p = self.torch_view[:, 0:3].view(self.n_robots, 3)
        self._q = self.torch_view[:, 3:7].view(self.n_robots, 4)
        self._v = self.torch_view[:, 7:10].view(self.n_robots, 3)
        self._omega = self.torch_view[:, 10:13].view(self.n_robots, 3)

        if self.gpu_mirror_exists():

            # gpu views
            self._p_gpu = self._gpu_mirror[:, 0:3].view(self.n_robots, 3)
            self._q_gpu = self._gpu_mirror[:, 3:7].view(self.n_robots, 4)
            self._v_gpu = self._gpu_mirror[:, 7:10].view(self.n_robots, 3)
            self._omega_gpu = self._gpu_mirror[:, 10:13].view(self.n_robots, 3)
    
    def _check_mirror_of_throw(self,
                        name: str):

        if not self.gpu_mirror_exists():

            exception = f"GPU mirror is not available!"

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def get_p(self,
            robot_idx: int = None,
            gpu: bool = False):
        
        if not gpu:
            
            if robot_idx is None:

                return self._p[:, :]
            
            else:

                return self._p[robot_idx, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_p")

            if robot_idx is None:

                return self._p_gpu[:, :]
            
            else:

                return self._p_gpu[robot_idx, :].view(1, -1)
            
    def get_q(self,
            robot_idx: int = None,
            gpu: bool = False):

        if self._q_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._q[:, :]
                
                else:

                    return self._q[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._q_gpu[:, :]
                
                else:

                    return self._q_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._q[:, self._q_remapping]
                
                else:

                    return self._q[robot_idx, self._q_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._q_gpu[:, self._q_remapping]
                
                else:

                    return self._q_gpu[robot_idx, self._q_remapping].view(1, -1)
                
    def get_v(self,
            robot_idx: int = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idx is None:

                return self._v[:, :]
            
            else:

                return self._v[robot_idx, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_v")

            if robot_idx is None:

                return self._v_gpu[:, :]
            
            else:

                return self._v_gpu[robot_idx, :].view(1, -1)
    
    def get_omega(self,
            robot_idx: int = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idx is None:

                return self._omega[:, :]
            
            else:

                return self._omega[robot_idx, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_omega")

            if robot_idx is None:

                return self._omega_gpu[:, :]
            
            else:

                return self._omega_gpu[robot_idx, :].view(1, -1)

class ContactWrenches(SharedDataView):

    def __init__(self,
            namespace = "",
            is_server = False, 
            n_robots: int = None, 
            n_contacts: int = None,
            contact_names: List[str] = None,
            contact_remapping: List[int] = None,
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = True):
        
        basename = "ContactWrenches"

        self.n_robots = n_robots
        self.n_contacts = n_contacts
        self.contact_names = contact_names

        if is_server:

            self.shared_contact_names = StringTensorServer(length = self.n_contacts, 
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)

        else:

            self.shared_contact_names = StringTensorClient(
                                        basename = basename + "Names", 
                                        name_space = namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
            
        n_cols = self.n_contacts * 6 # cart. force + torques

        self._contact_remapping = None
        if contact_remapping is not None:
            self._contact_remapping = torch.tensor(contact_remapping)

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_robots, 
            n_cols = n_cols, 
            dtype = sharsor_dtype.Float,
            verbose = verbose, 
            vlevel = vlevel,
            fill_value = 0, 
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=with_gpu_mirror)

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
                    
            written = self.shared_contact_names.write_vec(self.contact_names, 0)

            if not written:
                
                exception = "Could not write contact names on shared memory!"

                Journal.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                        
        else:
            
            self.contact_names = [""] * self.n_contacts

            written = self.shared_contact_names.read_vec(self.contact_names, 0)

            if not written:

                exception = "Could not read contact names on shared memory!"

                Journal.log(self.__class__.__name__,
                    name,
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
    def _init_views(self):

        # root
        self._f = self.torch_view[:, 0:self.n_contacts * 3].view(self.n_robots, 
                                                                self.n_contacts * 3)
        
        self._t = self.torch_view[:, (self.n_contacts * 3):(self.n_contacts * 6)].view(self.n_robots, 
                                                                self.n_contacts * 3)
        
        self._w = self.torch_view[:, :].view(self.n_robots, self.n_contacts * 6)

        if self.gpu_mirror_exists():

            self._f_gpu = self._gpu_mirror[:, 0:self.n_contacts * 3].view(self.n_robots, 
                                                                self.n_contacts * 3)
            self._t_gpu = self._gpu_mirror[:, self.n_contacts * 3:self.n_contacts * 6].view(self.n_robots, 
                                                                    self.n_contacts * 3)
            self._w_gpu = self._gpu_mirror[:, :].view(self.n_robots, self.n_contacts * 6)
    
    def get_f(self,
            robot_idx: int = None,
            gpu: bool = False):

        if self._contact_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._f[:, :]
                
                else:

                    return self._f[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_f")

                if robot_idx is None:

                    return self._f_gpu[:, :]
                
                else:

                    return self._f_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._f[:, self._contact_remapping]
                
                else:

                    return self._f[robot_idx, self._contact_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._f_gpu[:, self._contact_remapping]
                
                else:

                    return self._f_gpu[robot_idx, self._contact_remapping].view(1, -1)

    def get_t(self,
        robot_idx: int = None,
        gpu: bool = False):

        if self._contact_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._t[:, :]
                
                else:

                    return self._t[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_t")

                if robot_idx is None:

                    return self._t_gpu[:, :]
                
                else:

                    return self._t_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._t[:, self._contact_remapping]
                
                else:

                    return self._t[robot_idx, self._contact_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._t_gpu[:, self._contact_remapping]
                
                else:

                    return self._t_gpu[robot_idx, self._contact_remapping].view(1, -1)
    
    def get_f_contact(self,
            contact_name: str,
            robot_idx: int = None,
            gpu: bool = False):
        
        if not contact_name in self.contact_names:
            
            contact_list = "\t".join(self.contact_names)

            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"

            Journal.log(self.__class__.__name__,
                "get_f_contact",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        index = self.contact_names.index(contact_name)

        return self.get_f(robot_idx=robot_idx,
                    gpu=gpu)[:, (index * 3):((index+1) * 3)]

    def get_t_contact(self,
            contact_name: str,
            robot_idx: int = None,
            gpu: bool = False):

        if not contact_name in self.contact_names:
            
            contact_list = "\t".join(self.contact_names)

            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"

            Journal.log(self.__class__.__name__,
                "get_t_contact",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        index = self.contact_names.index(contact_name)

        return self.get_t(robot_idx=robot_idx,
                    gpu=gpu)[:, (index * 3):((index+1) * 3)]

    def get_w(self,
        robot_idx: int = None,
        gpu: bool = False):

        if self._contact_remapping is None:

            if not gpu:
                
                if robot_idx is None:

                    return self._w[:, :]
                
                else:

                    return self._w[robot_idx, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_t")

                if robot_idx is None:

                    return self._w_gpu[:, :]
                
                else:

                    return self._w_gpu[robot_idx, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idx is None:

                    return self._w[:, self._contact_remapping]
                
                else:

                    return self._w[robot_idx, self._contact_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idx is None:

                    return self._w_gpu[:, self._contact_remapping]
                
                else:

                    return self._w_gpu[robot_idx, self._contact_remapping].view(1, -1)

    def _check_mirror_of_throw(self,
                        name: str):

        if not self.gpu_mirror_exists():

            exception = f"GPU mirror is not available!"

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

class FullRobState(SharedDataBase):

    def __init__(self,
            namespace: str,
            basename: str,
            is_server: bool,
            n_robots: int = None,
            n_jnts: int = None,
            n_contacts: int = 1,
            jnt_names: List[str] = None,
            jnts_remapping: List[int] = None,
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            contact_remapping: List[int] = None, 
            with_gpu_mirror: bool = True,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1):

        self._namespace = namespace
        self._basename = basename

        self._is_server = is_server

        self._torch_device = torch.device("cuda")
        self._dtype = torch.float32

        self._verbose = verbose
        self._vlevel = vlevel

        self._n_robots = n_robots
        self._n_jnts = n_jnts
        self._n_contacts = n_contacts
        self._jnt_names = jnt_names
        self._contact_names = contact_names
        
        self._jnts_remapping = jnts_remapping
        self._q_remapping = q_remapping
        self._contact_remapping = contact_remapping

        self._safe = safe
        self._force_reconnection = force_reconnection
        
        self._with_gpu_mirror = with_gpu_mirror

        self.root_state = RootState(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            q_remapping=self._q_remapping,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror)
    
        self.jnts_state = JntsState(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            n_jnts=self._n_jnts,
                            jnt_names=self._jnt_names,
                            jnts_remapping=self._jnts_remapping,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror)
        
        self.contact_wrenches = ContactWrenches(namespace=self._namespace + self._basename, 
                            is_server=self._is_server,
                            n_robots=self._n_robots,
                            n_contacts=self._n_contacts,
                            contact_names=self._contact_names,
                            contact_remapping=self._contact_remapping,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=self._safe,
                            force_reconnection=self._force_reconnection,
                            with_gpu_mirror=with_gpu_mirror)
        
        self._is_running = False
    
    def __del__(self):

        self.close()

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
    
    def run(self):

        self.root_state.run()

        self.jnts_state.run()

        self.contact_wrenches.run()

        if not self._is_server:

            self._n_robots = self.jnts_state.n_robots

            self._n_jnts = self.jnts_state.n_jnts

            self._n_contacts = self.contact_wrenches.n_contacts

            self._jnt_names = self.jnts_state.jnt_names

            self._contact_names = self.contact_wrenches.contact_names

        self._is_running = True
            
    def synch_mirror(self,
                from_gpu: bool):

        if self._with_gpu_mirror:
            
            if from_gpu:
                
                # synchs root_state and jnt_state (which will normally live on GPU)
                # with the shared state data using the aggregate view (normally on CPU)

                # this requires a couple of (not so nice) COPIES FROM GPU TO CPU
                
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

            torch.cuda.synchronize() # this way we ensure that after this the state on GPU
            # is fully updated
    
    def synch_from_shared_mem(self):

        # reads from shared mem
        self.root_state.synch_all(read = True, wait = True)
        self.jnts_state.synch_all(read = True, wait = True)
        self.contact_wrenches.synch_all(read = True, wait = True)

    def synch_to_shared_mem(self):

        # write to shared mem
        self.root_state.synch_all(read = False, wait = True)
        self.jnts_state.synch_all(read = False, wait = True)
        self.contact_wrenches.synch_all(read = False, wait = True)
        
    def close(self):

        self.root_state.close()
        self.jnts_state.close()
        self.contact_wrenches.close()
