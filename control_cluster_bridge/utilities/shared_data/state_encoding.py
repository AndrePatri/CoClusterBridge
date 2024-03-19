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
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            fill_value: float = 0.0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False):
        
        basename = "JntsState" 

        n_cols = None

        if n_jnts is not None:

            n_cols = 4 * n_jnts # jnts config., vel., acc., torques

        self.n_jnts = n_jnts
        self.n_robots = n_robots
        self.jnt_names = jnt_names

        self._jnts_remapping = None
        
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

                exception = f"Provided jnt remapping length {len(jnts_remapping)}" + \
                    f"does not match n. joints {self.n_jnts}"

                Journal.log(self.__class__.__name__,
                    "update_jnts_remapping",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            self._jnts_remapping = torch.tensor(jnts_remapping)
        
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

    def get_remapping(self):

        return self._jnts_remapping
    
    def set_q(self,
            q: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    self._q[:, :] = q
                
                else:

                    self._q[robot_idxs, :] = q
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    self._q_gpu[:, :] = q
                
                else:

                    self._q_gpu[robot_idxs, :] = q
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    self._q[:, self._jnts_remapping] = q
                
                else:

                    self._q[robot_idxs, self._jnts_remapping] = q
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    self._q_gpu[:, self._jnts_remapping] = q
                
                else:

                    self._q_gpu[robot_idxs, self._jnts_remapping] = q
                
    def get_q(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    return self._q[:, :]
                
                else:

                    return self._q[robot_idxs, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    return self._q_gpu[:, :]
                
                else:

                    return self._q_gpu[robot_idxs, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    return self._q[:, self._jnts_remapping]
                
                else:

                    return self._q[robot_idxs, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    return self._q_gpu[:, self._jnts_remapping]
                
                else:

                    return self._q_gpu[robot_idxs, self._jnts_remapping].view(1, -1)

    def set_v(self,
            v: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    self._v[:, :] = v
                
                else:

                    self._v[robot_idxs, :] = v
            
            else:

                self._check_mirror_of_throw("set_v")

                if robot_idxs is None:

                    self._v_gpu[:, :] = v
                
                else:

                    self._v_gpu[robot_idxs, :] = v
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    self._v[:, self._jnts_remapping] = v
                
                else:

                    self._v[robot_idxs, self._jnts_remapping] = v
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    self._v_gpu[:, self._jnts_remapping] = v
                
                else:

                    self._v_gpu[robot_idxs, self._jnts_remapping] = v
      
    def get_v(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    return self._v[:, :]
                
                else:

                    return self._v[robot_idxs, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idxs is None:

                    return self._v_gpu[:, :]
                
                else:

                    return self._v_gpu[robot_idxs, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    return self._v[:, self._jnts_remapping]
                
                else:

                    return self._v[robot_idxs, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idxs is None:

                    return self._v_gpu[:, self._jnts_remapping]
                
                else:

                    return self._v_gpu[robot_idxs, self._jnts_remapping].view(1, -1)

    def set_a(self,
            a: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    self._a[:, :] = a
                
                else:

                    self._a[robot_idxs, :] = a
            
            else:

                self._check_mirror_of_throw("set_a")

                if robot_idxs is None:

                    self._a_gpu[:, :] = a
                
                else:

                    self._a_gpu[robot_idxs, :] = a
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    self._a[:, self._jnts_remapping] = a
                
                else:

                    self._a[robot_idxs, self._jnts_remapping] = a
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    self._a_gpu[:, self._jnts_remapping] = a
                
                else:

                    self._a_gpu[robot_idxs, self._jnts_remapping] = a
      
    def get_a(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    return self._a[:, :]
                
                else:

                    return self._a[robot_idxs, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idxs is None:

                    return self._a_gpu[:, :]
                
                else:

                    return self._a_gpu[robot_idxs, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    return self._a[:, self._jnts_remapping]
                
                else:

                    return self._a[robot_idxs, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_v")

                if robot_idxs is None:

                    return self._a_gpu[:, self._jnts_remapping]
                
                else:

                    return self._a_gpu[robot_idxs, self._jnts_remapping].view(1, -1)

    def set_eff(self,
            eff: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    self._eff[:, :] = eff
                
                else:

                    self._eff[robot_idxs, :] = eff
            
            else:

                self._check_mirror_of_throw("set_eff")

                if robot_idxs is None:

                    self._eff_gpu[:, :] = eff
                
                else:

                    self._eff_gpu[robot_idxs, :] = eff
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    self._eff[:, self._jnts_remapping] = eff
                
                else:

                    self._eff[robot_idxs, self._jnts_remapping] = eff
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    self._eff_gpu[:, self._jnts_remapping] = eff
                
                else:

                    self._eff_gpu[robot_idxs, self._jnts_remapping] = eff
      
    def get_eff(self,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):

        if self._jnts_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    return self._eff[:, :]
                
                else:

                    return self._eff[robot_idxs, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_eff")

                if robot_idxs is None:

                    return self._eff_gpu[:, :]
                
                else:

                    return self._eff_gpu[robot_idxs, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    return self._eff[:, self._jnts_remapping]
                
                else:

                    return self._eff[robot_idxs, self._jnts_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_eff")

                if robot_idxs is None:

                    return self._eff_gpu[:, self._jnts_remapping]
                
                else:

                    return self._eff_gpu[robot_idxs, self._jnts_remapping].view(1, -1)
         
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
            with_gpu_mirror: bool = False,
            fill_value = 0):
        
        basename = "RootState" 

        n_cols = 13 # p, q, v, omega + 

        self.n_robots = n_robots

        self._q_remapping = None

        if q_remapping is not None:

            self.set_q_remapping(q_remapping)

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

        # views of the underlying memory view of the 
        # actual shared memory (crazy, eh?)

        # root
        self._p = None
        self._q = None
        self._v = None
        self._omega = None
        self._q_full = None # full root configuration (pos + quaternion)
        self._v_full = None # full root velocity (lin. + angular)

        self._p_gpu = None
        self._q_gpu = None
        self._v_gpu = None
        self._omega_gpu = None
        self._q_full_gpu = None
        self._v_full_gpu = None 
        
    def run(self,
            q_remapping: List[int] = None):
        
        # overriding parent 

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")        
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
                    "update_jnts_remapping",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

            self._q_remapping = torch.tensor(q_remapping)

    def _init_views(self):

        # root
        self._p = self.torch_view[:, 0:3].view(self.n_robots, 3)
        self._q = self.torch_view[:, 3:7].view(self.n_robots, 4)
        self._q_full = self.torch_view[:, 0:7].view(self.n_robots, 7)

        self._v = self.torch_view[:, 7:10].view(self.n_robots, 3)
        self._omega = self.torch_view[:, 10:13].view(self.n_robots, 3)
        self._v_full = self.torch_view[:, 7:13].view(self.n_robots, 6)

        if self.gpu_mirror_exists():

            # gpu views
            self._p_gpu = self._gpu_mirror[:, 0:3].view(self.n_robots, 3)
            self._q_gpu = self._gpu_mirror[:, 3:7].view(self.n_robots, 4)
            self._q_full_gpu = self._gpu_mirror[:, 0:7].view(self.n_robots, 7)

            self._v_gpu = self._gpu_mirror[:, 7:10].view(self.n_robots, 3)
            self._omega_gpu = self._gpu_mirror[:, 10:13].view(self.n_robots, 3)
            self._v_full_gpu = self._gpu_mirror[:, 7:13].view(self.n_robots, 6)
    
    def _check_mirror_of_throw(self,
                        name: str):

        if not self.gpu_mirror_exists():

            exception = f"GPU mirror is not available!"

            Journal.log(self.__class__.__name__,
                name,
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def set_p(self,
            p: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):
        
        if not gpu:
            
            if robot_idxs is None:

                self._p[:, :] = p
            
            else:
                
                self._p[robot_idxs, :] = p
        
        else:

            self._check_mirror_of_throw("set_p")

            if robot_idxs is None:

                self._p_gpu[:, :] = p
            
            else:

                self._p_gpu[robot_idxs, :] = p
            
    def get_p(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):
        
        if not gpu:
            
            if robot_idxs is None:

                return self._p[:, :]
            
            else:

                return self._p[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_p")

            if robot_idxs is None:

                return self._p_gpu[:, :]
            
            else:

                return self._p_gpu[robot_idxs, :].view(1, -1)

    def set_q(self,
            q: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):
        
        if not gpu:
            
            if robot_idxs is None:

                self._q[:, :] = q
            
            else:

                self._q[robot_idxs, :] = q
        
        else:

            self._check_mirror_of_throw("set_q")

            if robot_idxs is None:

                self._q_gpu[:, :] = q
            
            else:

                self._q_gpu[robot_idxs, :] = q

    def get_q(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if self._q_remapping is None:

            if not gpu:
                
                if robot_idxs is None:

                    return self._q[:, :]
                
                else:

                    return self._q[robot_idxs, :].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    return self._q_gpu[:, :]
                
                else:

                    return self._q_gpu[robot_idxs, :].view(1, -1)
        
        else:

            if not gpu:
                
                if robot_idxs is None:

                    return self._q[:, self._q_remapping]
                
                else:

                    return self._q[robot_idxs, self._q_remapping].view(1, -1)
            
            else:

                self._check_mirror_of_throw("get_q")

                if robot_idxs is None:

                    return self._q_gpu[:, self._q_remapping]
                
                else:

                    return self._q_gpu[robot_idxs, self._q_remapping].view(1, -1)
    
    def set_q_full(self,
            q_full: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                self._q_full[:, :] = q_full
            
            else:
                
                self._q_full[robot_idxs, :] = q_full
        
        else:

            self._check_mirror_of_throw("set_q_full")

            if robot_idxs is None:

                self._q_full_gpu[:, :] = q_full
            
            else:

                self._q_full_gpu[robot_idxs, :] = q_full

    def get_q_full(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._q_full[:, :]
            
            else:

                return self._q_full[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_q_full")

            if robot_idxs is None:

                return self._q_full_gpu[:, :]
            
            else:

                return self._q_full_gpu[robot_idxs, :].view(1, -1)
            
    def set_v(self,
        v: torch.Tensor,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):
        
        if not gpu:
            
            if robot_idxs is None:

                self._v[:, :] = v
            
            else:

                self._v[robot_idxs, :] = v
        
        else:

            self._check_mirror_of_throw("set_v")

            if robot_idxs is None:

                self._v_gpu[:, :] = v
            
            else:

                self._v_gpu[robot_idxs, :] = v

    def get_v(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._v[:, :]
            
            else:

                return self._v[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_v")

            if robot_idxs is None:

                return self._v_gpu[:, :]
            
            else:

                return self._v_gpu[robot_idxs, :].view(1, -1)
    
    def set_omega(self,
        omega: torch.Tensor,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):
        
        if not gpu:
            
            if robot_idxs is None:

                self._omega[:, :] = omega
            
            else:

                self._omega[robot_idxs, :] = omega
        
        else:

            self._check_mirror_of_throw("set_omega")

            if robot_idxs is None:

                self._omega_gpu[:, :] = omega
            
            else:

                self._omega_gpu[robot_idxs, :] = omega

    def get_omega(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._omega[:, :]
            
            else:

                return self._omega[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_omega")

            if robot_idxs is None:

                return self._omega_gpu[:, :]
            
            else:

                return self._omega_gpu[robot_idxs, :].view(1, -1)

    def set_v_full(self,
        v_full: torch.Tensor,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                self._v_full[:, :] = v_full
            
            else:

                self._v_full[robot_idxs, :] = v_full
        
        else:

            self._check_mirror_of_throw("set_v_full")

            if robot_idxs is None:

                self._v_full_gpu[:, :] = v_full
            
            else:

                self._v_full_gpu[robot_idxs, :] = v_full

    def get_v_full(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._v_full[:, :]
            
            else:

                return self._v_full[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_v_full")

            if robot_idxs is None:

                return self._v_full_gpu[:, :]
            
            else:

                return self._v_full_gpu[robot_idxs, :].view(1, -1)
            
class ContactWrenches(SharedDataView):

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

            while not self.shared_contact_names.read_vec(self.contact_names, 0):

                Journal.log(self.__class__.__name__,
                        "run",
                        "Could not read contact names on shared memory. Retrying...",
                        LogType.WARN,
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
    
    def set_f(self,
            f: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            contact_idx: int = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:
                
                if contact_idx is None:

                    self._f[:, :] = f
                
                else:

                    self._f[:, (contact_idx * 3):((contact_idx+1) * 3)] = f
            
            else:

                if contact_idx is None:

                    self._f[robot_idxs, :] = f
                
                else:

                    self._f[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)] = f
        
        else:

            self._check_mirror_of_throw("set_f")

            if robot_idxs is None:
                
                if contact_idx is None:

                    self._f_gpu[:, :] = f
                
                else:

                    self._f_gpu[:, (contact_idx * 3):((contact_idx+1) * 3)] = f
            
            else:

                if contact_idx is None:

                    self._f_gpu[robot_idxs, :] = f
                
                else:
                    
                    self._f_gpu[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)] = f

    def get_f(self,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._f[:, :]
            
            else:

                return self._f[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_f")

            if robot_idxs is None:

                return self._f_gpu[:, :]
            
            else:

                return self._f_gpu[robot_idxs, :].view(1, -1)

    def set_t(self,
            t: torch.Tensor,
            robot_idxs: torch.Tensor = None,
            contact_idx: int = None,
            gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:
                
                if contact_idx is None:

                    self._t[:, :] = t
                
                else:

                    self._t[:, (contact_idx * 3):((contact_idx+1) * 3)] = t
            
            else:

                if contact_idx is None:

                    self._t[robot_idxs, :] = t
                
                else:

                    self._t[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)] = t
        
        else:

            self._check_mirror_of_throw("set_t")

            if robot_idxs is None:
                
                if contact_idx is None:

                    self._t_gpu[:, :] = t
                
                else:

                    self._t_gpu[:, (contact_idx * 3):((contact_idx+1) * 3)] = t
            
            else:

                if contact_idx is None:

                    self._t_gpu[robot_idxs, :] = t
                
                else:
                    
                    self._t_gpu[robot_idxs, (contact_idx * 3):((contact_idx+1) * 3)] = t

    def get_t(self,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._t[:, :]
            
            else:

                return self._t[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_t")

            if robot_idxs is None:

                return self._t_gpu[:, :]
            
            else:

                return self._t_gpu[robot_idxs, :].view(1, -1)
    
    def set_f_contact(self,
            f: torch.Tensor,
            contact_name: str,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):
        
        if not contact_name in self.contact_names:
            
            contact_list = "\t".join(self.contact_names)

            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"

            Journal.log(self.__class__.__name__,
                "set_f_contact",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        index = self.contact_names.index(contact_name)

        self.set_f(f =f,
            contact_idx=index,
            robot_idxs=robot_idxs,
            gpu=gpu)
                    
    def get_f_contact(self,
            contact_name: str,
            robot_idxs: torch.Tensor = None,
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

        return self.get_f(robot_idxs=robot_idxs,
                    gpu=gpu)[:, (index * 3):((index+1) * 3)]

    def set_t_contact(self,
            t: torch.Tensor,
            contact_name: str,
            robot_idxs: torch.Tensor = None,
            gpu: bool = False):
        
        if not contact_name in self.contact_names:
            
            contact_list = "\t".join(self.contact_names)

            exception = f"Contact name {contact_name} not in contact list [{contact_list}]"

            Journal.log(self.__class__.__name__,
                "set_t_contact",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        index = self.contact_names.index(contact_name)

        self.set_t(t =t,
            contact_idx=index,
            robot_idxs=robot_idxs,
            gpu=gpu)
        
    def get_t_contact(self,
            contact_name: str,
            robot_idxs: torch.Tensor = None,
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

        return self.get_t(robot_idxs=robot_idxs,
                    gpu=gpu)[:, (index * 3):((index+1) * 3)]

    def get_w(self,
        robot_idxs: torch.Tensor = None,
        gpu: bool = False):

        if not gpu:
            
            if robot_idxs is None:

                return self._w[:, :]
            
            else:

                return self._w[robot_idxs, :].view(1, -1)
        
        else:

            self._check_mirror_of_throw("get_t")

            if robot_idxs is None:

                return self._w_gpu[:, :]
            
            else:

                return self._w_gpu[robot_idxs, :].view(1, -1)

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
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            with_gpu_mirror: bool = False,
            force_reconnection: bool = False,
            safe: bool = True,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            fill_value = 0):

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
        
        self._jnts_remapping = None
        self._q_remapping = q_remapping

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
                            with_gpu_mirror=with_gpu_mirror,
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
                            fill_value=fill_value)
        
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

            #torch.cuda.synchronize() # this way we ensure that after this the state on GPU
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
