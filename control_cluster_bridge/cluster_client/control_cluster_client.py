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
from abc import ABC, abstractmethod

from control_cluster_bridge.utilities.cpu_utils.core_utils import get_isolated_cores

from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling

from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import VLevel

import psutil

from typing import List

import multiprocess as mp
import os

from perf_sleep.pyperfsleep import PerfSleep

from typing import TypeVar

ClusterSrvrChild = TypeVar('ClusterSrvrChild', bound='ControlClusterClient')

class ControlClusterClient(ABC):

    # This is meant to handle a cluster of controllers for a type of robot, 
    # where each controller is the same 
    # Different controllers (e.g. with different formulations, etc..) should
    # be handled by a separate cluster client 

    def __init__(self, 
            namespace: str,
            cluster_size: int,
            processes_basename: str = "Controller", 
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

        self.cluster_size = cluster_size

        self._device = "cpu"

        self._controllers: List[RHChild] = [] # list of controllers (must inherit from
        # RHController)

        # self.handshake_srvr = HanshakeDataCntrlSrvr(verbose=self.verbose, 
        #                                 namespace=self.namespace)
        # self.handshake_srvr.handshake()
        self._processes: List[mp.Process] = [] 

        self._is_cluster_ready = False

        self.pre_init_done = False
                 
        self.jnt_names_rhc = None # publishes joint names from controller to shared mem

        # shared memory 

        self.cluster_stats = None

        self._perf_timer = PerfSleep()

        self._terminated = False
    
    def __del__(self):

        if not self._terminated:

            self.terminate()
    
    def pre_init(self):
        
        # to be called before controllers are created/added

        Journal.log(self.__class__.__name__,
                        "pre_init",
                        "Performing pre-initialization steps...",
                        LogType.STAT,
                        throw_when_excep = True)

        self.cluster_stats = RhcProfiling(is_server=False, 
                                    name=self.namespace,
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

    def _spawn_controller(self,
                    idx: int,
                    core_idxs: List[int]):
        
        # this is run in a child process for each controller
        
        controller = self._generate_controller(idx=idx)

        controller.set_affinity(core_idx=self._compute_process_affinity(idx, core_ids=core_idxs))

        controller.solve() # runs the solution loop

    def run(self):
        
        if not self.pre_init_done:

            Journal.log(self.__class__.__name__,
                        "launch_controller",
                        f"pre_init() method has not been called yet!",
                        LogType.EXCEP,
                        throw_when_excep = True)
            
        self._spawn_processes()

        while True:

            try:

                nsecs = int(0.1 * 1e9)
                self._perf_timer.thread_sleep(nsecs) 

                continue

            except KeyboardInterrupt:

                self.terminate() # closes all processes
                
                break

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

        self._terminated = True
    
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
                        LogType.EXCEP,
                        throw_when_excep = False)

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
            
    def _compute_process_affinity(self, 
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
            
        self._debug_prints() # some debug prints
        
        core_ids = []
        if self.core_ids_override_list is None:
            core_ids = self._get_cores() # gets cores over which processes are to be distributed
        else:
            # ini case user wants to set core ids manually
            core_ids = self.core_ids_override_list

        for i in range(0, self.cluster_size):
            
            info = f"Spawning process for controller n.{i}."

            Journal.log(self.__class__.__name__,
                    "_spawn_processes",
                    info,
                    LogType.STAT,
                    throw_when_excep = True)

            process = mp.Process(target=self._spawn_controller, 
                            name=self.processes_basename + str(i),
                            args=(i, core_ids))
                        
            self._processes.append(process)

            self._processes[i].start()
            
        self._is_cluster_ready = True

        self.cluster_stats.write_info(dyn_info_name="cluster_ready",
                                    val=self._is_cluster_ready)

        Journal.log(self.__class__.__name__,
                    "_spawn_processes",
                    "Processes spawned.",
                    LogType.STAT,
                    throw_when_excep = True)

    @abstractmethod
    def _generate_controller(self,
                        idx: int):

        # to be overridden
        return None