from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import List

# Joint impedance control debug data

class JntImpCntrlData(SharedDataBase):

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

            jnt_names_read = self.shared_jnt_names.read_vec(self.jnt_names, 0)

            if not jnt_names_read:
                
                exception = f"Could not read joint names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run",
                    exception,
                    LogType.EXCEP,
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