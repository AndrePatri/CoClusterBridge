# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
from abc import ABC

from control_cluster_bridge.controllers.rhc import RHChild
from control_cluster_bridge.utilities.control_cluster_defs import HanshakeDataCntrlSrvr

from control_cluster_bridge.utilities.cpu_utils.core_utils import get_isolated_cores

from control_cluster_bridge.utilities.defs import jnt_names_rhc_name
from control_cluster_bridge.utilities.shared_mem import SharedStringArray

from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling

from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import VLevel

import psutil

from typing import List

import multiprocess as mp
import os

from typing import TypeVar

ClusterSrvrChild = TypeVar('ClusterSrvrChild', bound='ControlClusterSrvr')

class ControlClusterSrvr(ABC):

    def __init__(self, 
            namespace: str = "",
            processes_basename: str = "controller", 
            isolated_cores_only: bool = False,
            use_only_physical_cores: bool = False,
            core_ids_override_list: List[int] = None,
            verbose: bool = False):

        # ciao :D
        #        CR 

        self.isolated_cores_only = isolated_cores_only # will spawn each controller
        # in a isolated core, if they fit

        self.use_only_physical_cores = use_only_physical_cores

        self.core_ids_override_list = core_ids_override_list

        self.isolated_cores = []

        self.namespace = namespace
        
        self.verbose = verbose

        self.processes_basename = processes_basename

        self.cluster_size = -1

        self._device = "cpu"

        self._controllers: List[RHChild] = [] # list of controllers (must inherit from
        # RHController)

        self.handshake_srvr = HanshakeDataCntrlSrvr(verbose=self.verbose, 
                                        namespace=self.namespace)
        self.handshake_srvr.handshake()
        self.cluster_size = self.handshake_srvr.cluster_size.tensor_view[0, 0].item()

        self._processes: List[mp.Process] = [] 

        self._is_cluster_ready = False

        self.pre_init_done = False
         
        self._controllers_count = 0
        
        self.jnt_names_rhc = None # publishes joint names from controller to shared mem

        # shared memory 

        self.cluster_stats = None
    
    def __del__(self):

        self.terminate()
        
    def _close_processes(self):
    
        # Wait for each process to exit gracefully or terminate forcefully
                
        for process in self._processes:

            process.join(timeout=0.2)  # Wait for 5 seconds for each process to exit gracefully

            if process.is_alive():
                
                process.terminate()  # Forcefully terminate the process
            
            Journal.log(self.__class__.__name__,
                    "_close_processes",
                    "Terminating child process " + str(process.name),
                    LogType.STAT)
    
    def _close_shared_mem(self):

        if self.cluster_stats is not None:

            self.cluster_stats.close()

    def _get_cores(self):

        cores = None

        if not self.isolated_cores_only and not self.use_only_physical_cores:

            # distribute processes over all system cores and
            # over both physical and virtual cores

            cores = list(range(psutil.cpu_count()))

        elif not self.isolated_cores_only and self.use_only_physical_cores:

            # distribute processes over all physical cores

            physical_cores_n = psutil.cpu_count(logical=False)

            all_cores_n = psutil.cpu_count()

            all_cores = list(range(all_cores_n))

            core_ratio = all_cores_n/physical_cores_n

            cores = []

            for i in range(0, all_cores_n):
                
                if (i % core_ratio) == 0:

                    cores.append(i)
            
        elif self.isolated_cores_only and not self.use_only_physical_cores:

            # distribute processes over isolated cores only,
            # both physical and virtual
            cores = get_isolated_cores()[1]

        elif self.isolated_cores_only and self.use_only_physical_cores:
            # distribute processes over isolated and physical
            # cores only

            physical_cores_n = psutil.cpu_count(logical=False)
            all_cores_n = psutil.cpu_count()
            core_ratio = all_cores_n/physical_cores_n

            isolated_cores = get_isolated_cores()[1]

            isolated_cores_n = len(isolated_cores)

            cores = []

            for i in range(0, isolated_cores_n):
                
                if (i % core_ratio) == 0:

                    cores.append(isolated_cores[i])

        else:

            Journal.log(self.__class__.__name__,
                "_get_cores",
                "Invalid combination of flags for core distribution",
                LogType.EXCEP,
                throw_when_excep = True)

        return cores
 
    def _debug_prints(self):
            
        if self.core_ids_override_list is not None:
            
            # we distribute the controllers over the available ones
            warning = "Custom core id list provided. Will distribute controllers over those idxs."
            
            Journal.log(self.__class__.__name__,
                        "_debug_prints",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
            
        else:

            if self.isolated_cores_only:

                self.isolated_cores = get_isolated_cores()[1] # available isolated
                # cores -> if possible we spawn a controller for each isolated 
                # core

                if not len(self.isolated_cores) > 0: 
                    
                    exception ="No isolated cores found on this machine. Either isolate some cores or " + \
                        "deactivate the use_isolated_cores flag."
                    
                    Journal.log(self.__class__.__name__,
                        "_debug_prints",
                        exception,
                        LogType.EXEP,
                        throw_when_excep = True)

                    raise Exception(exception)
                        
                # we distribute the controllers over the available ones
                warning = "Will distribute controllers over physical cores only"
                
                Journal.log(self.__class__.__name__,
                            "_debug_prints",
                            warning,
                            LogType.WARN,
                            throw_when_excep = True)
                
            if self.isolated_cores_only:
                
                # we distribute the controllers over the available ones
                warning = "Will distribute controllers over isolated cores only"
                
                Journal.log(self.__class__.__name__,
                            "_debug_prints",
                            warning,
                            LogType.WARN,
                            throw_when_excep = True)
            
        if len(self.isolated_cores) < self.cluster_size and self.isolated_cores_only:
            
            # we distribute the controllers over the available ones
            warning = "Not enough isolated cores available to distribute the controllers " + \
                f"on them. N. available isolated cores: {len(self.isolated_cores)}, n. controllers {self.cluster_size}. "+ \
                "Processes will be distributed among the available ones."
            
            Journal.log(self.__class__.__name__,
                        "_debug_prints",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
            
        if len(self.isolated_cores) < self.cluster_size and self.isolated_cores_only:
            
            # we distribute the controllers over the available ones
            warning = "Not enough isolated cores available to distribute the controllers " + \
                f"on them. N. available isolated cores: {len(self.isolated_cores)}, n. controllers {self.cluster_size}. "+ \
                "Processes will be distributed among the available ones."
            
            Journal.log(self.__class__.__name__,
                        "_debug_prints",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
            
    def _get_process_affinity(self, 
                        process_index: int, 
                        core_ids: List[int]):

        num_cores = len(core_ids)

        return core_ids[process_index % num_cores]

    def _spawn_processes(self):
        
        Journal.log(self.__class__.__name__,
                        "_spawn_processes",
                        "spawning processes...",
                        LogType.STAT,
                        throw_when_excep = True)

        if self._controllers_count == self.cluster_size:
            
            self._debug_prints() # some debug prints

            for i in range(0, self.cluster_size):

                process = mp.Process(target=self._controllers[i].solve, name=self.processes_basename + str(i))

                self._processes.append(process)
            
            core_ids = []

            if self.core_ids_override_list is None:

                core_ids = self._get_cores() # gets cores over which processes are to be distributed

            else:
                
                # ini case user wants to set core ids manually
                core_ids = self.core_ids_override_list

            i = 0
            for process in self._processes:

                process.start()

                os.sched_setaffinity(process.pid, {self._get_process_affinity(i, core_ids=core_ids)})
                
                info = f"Setting affinity ID {os.sched_getaffinity(process.pid)} for controller n.{i}."

                Journal.log(self.__class__.__name__,
                        "_spawn_processes",
                        info,
                        LogType.STAT,
                        throw_when_excep = True)
                
                i += 1

            self._is_cluster_ready = True

            self.cluster_stats.write_info(dyn_info_name="cluster_ready",
                                        val=self._is_cluster_ready)

            Journal.log(self.__class__.__name__,
                        "_spawn_processes",
                        "Processes spawned.",
                        LogType.STAT,
                        throw_when_excep = True)
                            
        else:
            
            Journal.log(self.__class__.__name__,
                        "_spawn_processes",
                        "You didn't finish to fill the cluster. Please call the add_controller() method to do so.",
                        LogType.EXCEP,
                        throw_when_excep = True)

    def pre_init(self):
        
        # to be called before controllers are created/added

        Journal.log(self.__class__.__name__,
                        "pre_init",
                        "Performing pre-initialization steps...",
                        LogType.STAT,
                        throw_when_excep = True)
        
        cluster_info_dict = {}
        cluster_info_dict["cluster_size"] = self.cluster_size

        self.cluster_stats = RhcProfiling(cluster_size=self.cluster_size,
                                        is_server=True, 
                                        name=self.namespace,
                                        param_dict=cluster_info_dict,
                                        verbose=self.verbose,
                                        vlevel=VLevel.V2,
                                        safe=True)
        
        self.cluster_stats.run()

        self.pre_init_done = True

        Journal.log(self.__class__.__name__,
                        "pre_init",
                        "Performing pre-initialization steps...",
                        LogType.STAT,
                        throw_when_excep = True)

    def _finalize_init(self):

        # steps to be performed after the controllers are fully initialized 

        Journal.log(self.__class__.__name__,
                        "_finalize_init",
                        "Performing final initialization steps...",
                        LogType.STAT,
                        throw_when_excep = True)
        
        self.handshake_srvr.finalize_init(add_data_length=self._controllers[0].add_data_lenght, 
                                    n_contacts=self._controllers[0].n_contacts)
        
        for i in range(0, self.cluster_size):

            # we assign the client-side joint names to each controller (used for mapping purposes)
            self._controllers[i].assign_client_side_jnt_names(self.handshake_srvr.jnt_names_client.read())

            self._controllers[i].create_jnt_maps()

            self._controllers[i].init_states() # initializes states

            self._controllers[i].set_cmds_to_homing() # safe cmds

            self._controllers[i].init_rhc_task_cmds() # initializes rhc commands

        # publishing joint names on shared memory for external use
        self.jnt_names_rhc = SharedStringArray(length=len(self._controllers[0].get_server_side_jnt_names()), 
                                name=jnt_names_rhc_name(), 
                                namespace=self.namespace,
                                is_server=True)

        self.jnt_names_rhc.start(init = self._controllers[0].get_server_side_jnt_names())
        
        Journal.log(self.__class__.__name__,
                        "_finalize_init",
                        "Final initialization steps completed.",
                        LogType.STAT,
                        throw_when_excep = True)
        
    def add_controller(self, controller: RHChild):
        
        if not self.pre_init_done:

            Journal.log(self.__class__.__name__,
                        "add_controller",
                        f"pre_init() method has not been called yet!",
                        LogType.EXCEP,
                        throw_when_excep = True)
        
        if self._controllers_count < self.cluster_size:

            self._controllers.append(controller)
            
            self._controllers_count += 1

            if self._controllers_count == self.cluster_size:
            
                self._finalize_init()
        
            return True

        if self._controllers_count > self.cluster_size:
            
            Journal.log(self.__class__.__name__,
                        "_finalize_init",
                        "The cluster is full. Cannot add any more controllers.",
                        LogType.WARN,
                        throw_when_excep = True)
            
            return False
    
    def start(self):

        self._spawn_processes()

    def terminate(self):
        
        Journal.log(self.__class__.__name__,
                        "_finalize_init",
                        "terminating cluster...",
                        LogType.STAT,
                        throw_when_excep = True)
        
        if self.jnt_names_rhc is not None:

            self.jnt_names_rhc.terminate()

        self._close_processes() # we also terminate all the child processes

        self._close_shared_mem()
