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

import time 

from control_cluster_bridge.utilities.defs import trigger_flagname
from control_cluster_bridge.utilities.defs import Journal
from control_cluster_bridge.utilities.homing import RobotHomer

from typing import List, TypeVar

import torch

from perf_sleep.pyperfsleep import PerfSleep

class CntrlCmd(ABC):

    pass

class RHCCmd(ABC):

    pass

RHCCmdChild = TypeVar('RHCCmdChild', bound='RHCCmd')

class RHController(ABC):

    def __init__(self, 
            cluster_size: int,
            controller_index: int,
            srdf_path: str,
            verbose = False, 
            debug = False,
            array_dtype = torch.float32, 
            namespace = ""):
        
        self.namespace = namespace
        
        self.journal = Journal()

        self.perf_timer = PerfSleep()

        self.controller_index = controller_index

        self.srdf_path = srdf_path

        self._verbose = verbose
        self._debug = debug

        self.cluster_size = cluster_size

        self.n_dofs = None
        self.n_contacts = None
        
        # shared mem
        self.robot_state = None 
        self.trigger_flag = None
        self.robot_cmds = None
        self.rhc_task_refs:RhcTaskRefsChild = None

        # jnt names
        self._client_side_jnt_names = []
        self._server_side_jnt_names = []
        self._got_jnt_names_client = False
        self._got_jnt_names_server = False

        # data maps
        self._to_server = []
        self._to_client = []
        self._quat_remap = [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
        self._jnt_maps_created = False
        
        self._states_initialized = False

        self.array_dtype = array_dtype

        self.add_data_lenght = 0

        self._homer: RobotHomer = None

        self._init()

    def __del__(self):
        
        self._terminate()

    def _terminate(self):
        
        a = 1

    def _init(self):

        self._init_problem() # we call the child's initialization method

        self._homer = RobotHomer(self.srdf_path, 
                            self._server_side_jnt_names)
        
    def init_rhc_task_cmds(self):
        
        a = 2
        
    def init_states(self):
        
        a = 3 

        self._states_initialized = True
    
    def set_cmds_to_homing(self):
        
        a = 4

    def _fill_cmds_from_sol(self):

        # gets data from the solution and updates the view on the shared data
        
        a = 5
        
    def solve(self):
        
        if not self._jnt_maps_created:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.journal.exception}]" + f"[{self.solve.__name__}]" + \
                                ":" + f"jnt maps not initialized. Did you call the create_jnt_maps()?"

            raise Exception(exception)

        if not self._states_initialized:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                                f"[{self.journal.exception}]" + f"[{self.solve.__name__}]" + \
                                ":" + f"states not initialized. Did you call the init_states()?"

            raise Exception(exception)

        
        while True:
            
            # we are always listening for a trigger signal from the client 

            try:
            
                if True:
                    
                    if self._debug:
                        
                        start = time.perf_counter()

                    # latest state is employed

                    self._solve() # solve actual TO

                    self._fill_cmds_from_sol() # we upd update the views of the cmd
                    # from the solution

                    if self._debug:
                        
                        duration = time.perf_counter() - start

                    if self._verbose and self._debug:

                        print("[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                            f"[{self.journal.info}]" + ":" + f"solve loop execution time  -> " + str(duration))

                    time.sleep(0.05) # seconds
                    

            except KeyboardInterrupt:

                break
    
    def assign_client_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._client_side_jnt_names = jnt_names
        
        self._got_jnt_names_client = True

    def _assign_server_side_jnt_names(self, 
                        jnt_names: List[str]):

        self._server_side_jnt_names = jnt_names

        self._got_jnt_names_server = True

    def _check_jnt_names_compatibility(self):

        set_srvr = set(self._server_side_jnt_names)
        set_client  = set(self._client_side_jnt_names)

        if not set_srvr == set_client:

            exception = f"[{self.__class__.__name__}]" + f"{self.controller_index}" + f"{self.journal.exception}" + \
                ": server-side and client-side joint names do not match!"

            raise Exception(exception)
        
    def create_jnt_maps(self):

        self._check_jnt_names_compatibility() # will raise exception

        if not self._got_jnt_names_client:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                f"[{self.journal.exception}]" + ":" + f"Cannot run the solve().  assign_client_side_jnt_names() was not called!"

            raise Exception(exception)
        
        if not self._got_jnt_names_server:

            exception = "[" + self.__class__.__name__ + str(self.controller_index) + "]"  + \
                f"[{self.journal.exception}]" + ":" + f"Cannot run the solve().  _assign_server_side_jnt_names() was not called!"

            raise Exception(exception)
        
        self._to_server = [self._client_side_jnt_names.index(element) for element in self._server_side_jnt_names]

        self._to_client = [self._server_side_jnt_names.index(element) for element in self._client_side_jnt_names]
        
        self._jnt_maps_created = True

    @abstractmethod
    def _init_rhc_task_cmds(self):

        pass

    @abstractmethod
    def _get_robot_jnt_names(self) -> List[str]:

        pass

    @abstractmethod
    def _get_cmd_jnt_q_from_sol(self) -> torch.Tensor:

        pass

    @abstractmethod
    def _get_cmd_jnt_v_from_sol(self) -> torch.Tensor:

        pass
    
    @abstractmethod
    def _get_cmd_jnt_eff_from_sol(self) -> torch.Tensor:

        pass
    
    @abstractmethod
    def _get_additional_slvr_info(self) -> torch.Tensor:

        pass

    @abstractmethod
    def _update_open_loop(self):

        # updates measured robot state 
        # using the internal robot state of the RHC controller

        pass
    
    @abstractmethod
    def _update_closed_loop(self):

        # updates measured robot state 
        # using the provided measurements

        pass

    @abstractmethod
    def _solve(self):

        pass
            
    @abstractmethod
    def _get_ndofs(self):

        pass

    @abstractmethod
    def _init_problem(self):

        # initialized horizon's TO problem

        pass
   
RHChild = TypeVar('RHChild', bound='RHController')

