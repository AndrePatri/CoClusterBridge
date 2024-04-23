from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import List

import torch

# Joint impedance control debug data

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
                with_gpu_mirror: bool = False,
                with_torch_view: bool = False):
            
            basename = "ImpedanceData" # hardcoded
            
            # the way data is ordered is in a tensor of shape
            # [n_envs x (n_jns x n_fields)]
            # Along columns 

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = n_jnts * 11, 
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection=force_reconnection,
                safe=safe,
                with_torch_view = with_torch_view,
                with_gpu_mirror = with_gpu_mirror)

            self._pos_err = None
            self._vel_err = None
            self._pos_gains = None
            self._vel_gains = None
            self._eff_ff = None
            self._pos = None
            self._pos_ref = None
            self._vel = None
            self._vel_ref = None
            self._eff = None
            self._imp_eff = None
            
            self.n_jnts = None

        def _init_views(self):

            if self._with_torch_view:
                self._pos_err = self.get_torch_mirror()[:, 0:self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_err = self.get_torch_mirror()[:, 1*self.n_jnts:2*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos_gains = self.get_torch_mirror()[:, 2*self.n_jnts:3*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_gains = self.get_torch_mirror()[:, 3*self.n_jnts:4*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._eff_ff = self.get_torch_mirror()[:, 4*self.n_jnts:5*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos = self.get_torch_mirror()[:, 5*self.n_jnts:6*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos_ref = self.get_torch_mirror()[:, 6*self.n_jnts:7*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel = self.get_torch_mirror()[:, 7*self.n_jnts:8*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_ref = self.get_torch_mirror()[:, 8*self.n_jnts:9*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._eff = self.get_torch_mirror()[:, 9*self.n_jnts:10*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._imp_eff = self.get_torch_mirror()[:, 10*self.n_jnts:11*self.n_jnts].view(self.n_rows, self.n_jnts)
            else:
                self._pos_err = self.get_numpy_mirror()[:, 0:self.n_jnts].view()
                self._vel_err = self.get_numpy_mirror()[:, 1*self.n_jnts:2*self.n_jnts].view()
                self._pos_gains = self.get_numpy_mirror()[:, 2*self.n_jnts:3*self.n_jnts].view()
                self._vel_gains = self.get_numpy_mirror()[:, 3*self.n_jnts:4*self.n_jnts].view()
                self._eff_ff = self.get_numpy_mirror()[:, 4*self.n_jnts:5*self.n_jnts].view()
                self._pos = self.get_numpy_mirror()[:, 5*self.n_jnts:6*self.n_jnts].view()
                self._pos_ref = self.get_numpy_mirror()[:, 6*self.n_jnts:7*self.n_jnts].view()
                self._vel = self.get_numpy_mirror()[:, 7*self.n_jnts:8*self.n_jnts].view()
                self._vel_ref = self.get_numpy_mirror()[:, 8*self.n_jnts:9*self.n_jnts].view()
                self._eff = self.get_numpy_mirror()[:, 9*self.n_jnts:10*self.n_jnts].view()
                self._imp_eff = self.get_numpy_mirror()[:, 10*self.n_jnts:11*self.n_jnts].view()
                self._p = self.get_numpy_mirror()[:, 0:3].view()
            if self.gpu_mirror_exists():
                # gpu views
                self._pos_err_gpu = self._gpu_mirror[:, 0:self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_err_gpu = self._gpu_mirror[:, 1*self.n_jnts:2*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos_gains_gpu = self._gpu_mirror[:, 2*self.n_jnts:3*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_gains_gpu = self._gpu_mirror[:, 3*self.n_jnts:4*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._eff_ff_gpu = self._gpu_mirror[:, 4*self.n_jnts:5*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos_gpu = self._gpu_mirror[:, 5*self.n_jnts:6*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._pos_ref_gpu = self._gpu_mirror[:, 6*self.n_jnts:7*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_gpu = self._gpu_mirror[:, 7*self.n_jnts:8*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._vel_ref_gpu = self._gpu_mirror[:, 8*self.n_jnts:9*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._eff_gpu = self._gpu_mirror[:, 9*self.n_jnts:10*self.n_jnts].view(self.n_rows, self.n_jnts)
                self._imp_eff_gpu = self._gpu_mirror[:, 10*self.n_jnts:11*self.n_jnts].view(self.n_rows, self.n_jnts)

        def _retrieve_data(self,
                name: str,
                gpu: bool = False):
        
            if not gpu:
                if name == "pos_err":
                    return self._pos_err
                if name == "vel_err":
                    return self._vel_err
                if name == "pos_gains":
                    return self._pos_gains
                if name == "vel_gains":
                    return self._vel_gains
                if name == "eff_ff":
                    return self._eff_ff
                if name == "pos":
                    return self._pos
                if name == "pos_ref":
                    return self._pos_ref
                if name == "vel":
                    return self._vel
                if name == "vel_ref":
                    return self._vel_ref
                if name == "eff":
                    return self._eff
                if name == "imp_eff":
                    return self._imp_eff
                else:
                    return None
            else:
                if name == "pos_err":
                    return self._pos_err_gpu
                if name == "vel_err":
                    return self._vel_err_gpu
                if name == "pos_gains":
                    return self._pos_gains_gpu
                if name == "vel_gains":
                    return self._vel_gains_gpu
                if name == "eff_ff":
                    return self._eff_ff_gpu
                if name == "pos":
                    return self._pos_gpu
                if name == "pos_ref":
                    return self._pos_ref_gpu
                if name == "vel":
                    return self._vel_gpu
                if name == "vel_ref":
                    return self._vel_ref_gpu
                if name == "eff":
                    return self._eff_gpu
                if name == "imp_eff":
                    return self._imp_eff_gpu
                else:
                    return None
            
        def run(self):
            super().run()
            self.n_jnts = int(self.n_cols / 11)
            self._init_views()

        def set(self,
            data,
            data_type: str,
            robot_idxs= None,
            gpu: bool = False):

            internal_data = self._retrieve_data(name=data_type,
                    gpu=gpu)
            if robot_idxs is None:
                internal_data[:, :] = data
            else:
                internal_data[robot_idxs, :] = data

        def get(self,
            data_type: str,
            robot_idxs = None,
            gpu: bool = False):

            internal_data = self._retrieve_data(name=data_type,
                        gpu=gpu)
                
            if robot_idxs is None:
                return internal_data
            else:
                return internal_data[robot_idxs, :]

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
        with_torch_view = True if self._use_gpu else False
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
                                    with_gpu_mirror=self._use_gpu,
                                    with_torch_view=with_torch_view)

        self._is_runnning = False

    def __del__(self):

        self.close()

    def is_running(self):
        
        return self._is_runnning
    
    def get_shared_mem(self):
        return [self.shared_jnt_names.get_shared_mem(),
            self.imp_data_view.get_shared_mem()]
    
    def run(self):
        
        self.imp_data_view.run()

        # in case we are clients
        self.n_envs = self.imp_data_view.n_rows
        self.n_jnts = self.imp_data_view.n_jnts

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
                
                print(self.jnt_names)
                Journal.log(self.__class__.__name__,
                    "run",
                    "Could not read joint names on shared memory. Retrying...",
                    LogType.WARN,
                    throw_when_excep = True)
                
    def close(self):
        
        self.imp_data_view.close()
        self.shared_jnt_names.close()