from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import List

import torch

# Joint impedance control debug data
class JntImpCntrlDataOld(SharedDataBase):

    class PosErrView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "PosErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                safe=safe,
                with_torch_view = True,
                with_gpu_mirror = with_gpu_mirror)
    
    class VelErrView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "VelErr" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                safe=safe,
                with_torch_view = True,
                with_gpu_mirror = with_gpu_mirror)
    
    class PosGainsView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "PosGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)
    
    class VelGainsView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "VelGain" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class PosView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "Pos" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class VelView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "Vel" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class EffView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "Eff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class PosRefView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "PosRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class VelRefView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "VelRef" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class EffFFView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "EffFF" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    class ImpEffView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "ImpEff" # hardcoded

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                with_torch_view = True,
                safe=safe,
                with_gpu_mirror = with_gpu_mirror)

    def __init__(self, 
            is_server = False, 
            n_envs: int = -1, 
            n_jnts: int = -1,
            jnt_names: List[str] = [""],
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            safe: bool = True,
            use_gpu: bool = False):

        self.is_server = is_server

        self.n_envs = n_envs
        self.n_jnts = n_jnts

        self.jnt_names = jnt_names

        self.verbose = verbose
        self.vlevel = vlevel
        
        self._use_gpu = use_gpu
        if self.is_server:

            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = "JntImpCntrlSharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel,
                                        force_reconnection = force_reconnection,
                                        safe = safe)

        else:

            self.shared_jnt_names = StringTensorClient(
                                        basename = "JntImpCntrlSharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel,
                                        safe = safe)

        self.pos_err_view = self.PosErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)
        
        self.vel_err_view = self.VelErrView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)
        
        self.pos_gains_view = self.PosGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.vel_gains_view = self.VelGainsView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.eff_ff_view = self.EffFFView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.pos_view = self.PosView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)
        
        self.pos_ref_view = self.PosRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.vel_view = self.VelView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.vel_ref_view = self.VelRefView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.eff_view = self.EffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)

        self.imp_eff_view = self.ImpEffView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)
        
        self._is_runnning = False

    def __del__(self):

        self.terminate()

    def is_running(self):
        
        return self._is_runnning

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

            while not self.shared_jnt_names.read_vec(self.jnt_names, 0):

                Journal.log(self.__class__.__name__,
                        "run",
                        "Could not read joint names on shared memory. Retrying...",
                        LogType.WARN,
                        throw_when_excep = True)
                
    def close(self):
        
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

class JntImpCntrlData(SharedDataBase):

    class ImpDataView(SharedTWrapper):

        def __init__(self,
                namespace = "",
                is_server = False, 
                n_envs: int = -1, 
                n_jnts: int = -1, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                force_reconnection: bool = False,
                safe: bool = True,
                with_gpu_mirror: bool = False):
            
            basename = "ImpedanceData" # hardcoded
            
            # the way data is ordered is in a tensor of shape
            # [n_envs x (n_jns x n_fields)]
            # Along columns 
            self._fields = ["pos_err", "vel_err", "pos_gains", "vel_gains", "eff_ff", 
                "pos", "pos_ref", "vel", "vel_ref", 
                "eff", "imp_eff"]
            n_fields = len(self._fields)
            
            self._n_jnts = n_jnts

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts * n_fields, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                safe=safe,
                with_torch_view = True,
                with_gpu_mirror = with_gpu_mirror)

        def set(self,
            name: str,
            data: torch.Tensor):

            idxs = self._fields.index(name)

            imp_data = self.get_torch_view()

            start_idx = idxs * self._n_jnts
            imp_data[:, start_idx:(start_idx + self._n_jnts)] = data

    def __init__(self, 
            is_server = False, 
            n_envs: int = -1, 
            n_jnts: int = -1,
            jnt_names: List[str] = [""],
            namespace = "", 
            verbose = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False,
            safe: bool = True,
            use_gpu: bool = False):

        self.is_server = is_server

        self.n_envs = n_envs
        self.n_jnts = n_jnts

        self.jnt_names = jnt_names

        self.verbose = verbose
        self.vlevel = vlevel
        
        self._use_gpu = use_gpu
        if self.is_server:

            self.shared_jnt_names = StringTensorServer(length = self.n_jnts, 
                                        basename = "JntImpCntrlSharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel,
                                        force_reconnection = force_reconnection,
                                        safe = safe)

        else:

            self.shared_jnt_names = StringTensorClient(
                                        basename = "JntImpCntrlSharedJntNames", 
                                        name_space = namespace,
                                        verbose = self.verbose, 
                                        vlevel = self.vlevel,
                                        safe = safe)

        self.imp_data_view = self.ImpDataView(is_server = self.is_server, 
                                    n_envs = self.n_envs, 
                                    n_jnts = self.n_jnts,
                                    namespace = namespace, 
                                    verbose = self.verbose, 
                                    vlevel = self.vlevel,
                                    force_reconnection=force_reconnection,
                                    safe=safe,
                                    with_gpu_mirror=self._use_gpu)
        
        self._is_runnning = False

    def __del__(self):

        self.close()

    def is_running(self):
        
        return self._is_runnning

    def run(self):
        
        self.imp_data_view.run()

        # in case we are clients
        self.n_envs = self.imp_data_view.n_rows
        self.n_jnts = self.imp_data_view.n_cols

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
            while not self.shared_jnt_names.read_vec(self.jnt_names, 0):
                Journal.log(self.__class__.__name__,
                    "run",
                    "Could not read joint names on shared memory. Retrying...",
                    LogType.WARN,
                    throw_when_excep = True)
                
    def close(self):
        
        self.imp_data_view.close()
        self.shared_jnt_names.close()