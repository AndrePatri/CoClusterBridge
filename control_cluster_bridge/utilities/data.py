from SharsorIPCpp.PySharsorIPC import dtype

from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype 
from SharsorIPCpp.PySharsorIPC import Journal

import numpy as np

import torch

from typing import List

# Joint impedance control debug data

class JntImpCntrlData:

    class PosErrView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "PosErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)
    
    class VelErrView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "VelErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)
    
    class PosGainsView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "PosGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)
    
    class VelGainsView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "VelGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class PosView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "Pos" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class VelView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "Vel" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class EffView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "Eff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class PosRefView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "PosRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class VelRefView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "VelRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class EffFFView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "EffFF" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    class ImpEffView(SharedDataView):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False):
            
            basename = "ImpEff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection)

    def __init__(self, 
            is_server = False, 
            n_envs: int = -1, 
            n_jnts: int = -1,
            jnt_names: List[str] = [""],
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        self.is_server = is_server

        self.n_envs = n_envs
        self.n_jnts = n_jnts

        self.jnt_names = jnt_names

        self.verbose = verbose
        self.vlevel = vlevel

        if self.is_server:

            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = "SharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel)

        else:

            self.shared_jnt_names = StringTensorClient(
                                        basename = "SharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel)

        self.pos_err_view = self.PosErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)
        
        self.vel_err_view = self.VelErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)
        
        self.pos_gains_view = self.PosGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.vel_gains_view = self.VelGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.eff_ff_view = self.EffFFView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.pos_view = self.PosView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)
        
        self.pos_ref_view = self.PosRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.vel_view = self.VelView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.vel_ref_view = self.VelRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.eff_view = self.EffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

        self.imp_eff_view = self.ImpEffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection)

    def __del__(self):

        self.terminate()

    def run(self):
        
        self.pos_err_view.run()
        self.vel_err_view.run()
        self.pos_gains_view.run()
        self.vel_gains_view.run()
        self.eff_ff_view.run()
        self.pos_view.run()
        self.pos_ref_view.run()
        self.vel_view.run()
        self.vel_ref_view.run()
        self.eff_view.run()
        self.imp_eff_view.run()

        # in case we are clients
        self.n_envs = self.pos_err_view.n_rows
        self.n_jnts = self.pos_err_view.n_cols

        # retrieving joint names
        self.shared_jnt_names.run()

        if self.is_server:

            jnt_names_written = self.shared_jnt_names.write_vec(self.jnt_names, 0)

            if not jnt_names_written:
                
                exception = f"Could not write joint names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                        
        else:
            
            self.jnt_names = [""] * self.n_jnts

            jnt_names_read = self.shared_jnt_names.read_vec(self.jnt_names, 0)

            if not jnt_names_read:
                
                exception = f"Could not read joint names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
    def terminate(self):
        
        self.pos_err_view.close()
        self.vel_err_view.close()
        self.pos_gains_view.close()
        self.vel_gains_view.close()
        self.eff_ff_view.close()
        self.pos_view.close()
        self.pos_ref_view.close()
        self.vel_view.close()
        self.vel_ref_view.close()
        self.eff_view.close()
        self.imp_eff_view.close()

        self.shared_jnt_names.close()

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

class FullRobState():

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
        self.contact_wrenches.synch_all(read = True, wait = True)
        
    def close(self):

        self.root_state.close()
        self.jnts_state.close()
        self.contact_wrenches.close()

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
            jnts_remapping: List[int] = None,
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            contact_remapping: List[int] = None, 
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
            jnts_remapping=jnts_remapping,
            contact_names=contact_names,
            q_remapping=q_remapping,
            contact_remapping=contact_remapping, 
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
            jnts_remapping: List[int] = None,
            contact_names: List[str] = None,
            q_remapping: List[int] = None,
            contact_remapping: List[int] = None, 
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
            jnts_remapping=jnts_remapping,
            contact_names=contact_names,
            q_remapping=q_remapping,
            contact_remapping=contact_remapping, 
            with_gpu_mirror=with_gpu_mirror,
            force_reconnection=force_reconnection,
            safe=safe,
            verbose=verbose,
            vlevel=vlevel)

# receding horizon control data

class RHCStatus():
    
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
                safe = False, 
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
                safe = False, 
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
                safe = False, 
                dtype=dtype.Bool,
                force_reconnection=force_reconnection,
                fill_value = False)
            
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
        
    def __del__(self):

        self.close()

    def run(self):

        self.resets.run()
        self.trigger.run()
        self.fails.run()

    def close(self):
        
        self.trigger.close()
        self.resets.close()
        self.fails.close()    

class RHCInternal():

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

        exception = "RHCInternal not initialized. Did you call the run()?"

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
            
class RHCRefs():

    def __init__(self):

        a = 1

# cluster coordination data

class HandShaker():

    def __init__(self):

        a = 1

# other things
        
class RHC2SharedNamings:

    def __init__(self, 
            basename: str, 
            namespace: str, 
            index: int):

        self.index = index

        self.basename = basename
        self.namespace = namespace

        self.global_ns = f"{basename}_{namespace}_{self.index}"

        self.ROBOT_Q_NAME = "robot_q"
        self.RHC_Q_NAME = "rhc_q"

    def global_ns(self, 
            basename: str, 
            namespace: str):

        return f"{basename}_{namespace}"
    
    def get_robot_q_name(self):
        
        return f"{self.global_ns}_{self.ROBOT_Q_NAME}"
    
    def get_rhc_q_name(self):

        return f"{self.global_ns}_{self.RHC_Q_NAME}"