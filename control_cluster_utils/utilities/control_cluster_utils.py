import torch

from abc import ABC, abstractmethod

from control_cluster_utils.controllers.rhc import RHChild

import multiprocess as mp
import os
import struct

import time

from typing import TypeVar, List

import numpy as np

class RobotClusterState:

    class RootStates:

        def __init__(self, 
                    cluster_size: int = 1, 
                    device: str = "cpu", 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device

            self.p = torch.zeros((cluster_size, 3), device = self._device, dtype=self.dtype) # floating base positions
            self.q = torch.zeros((cluster_size, 4), device = self._device, dtype=self.dtype) # floating base orientation (quaternion)
            self.v = torch.zeros((cluster_size , 3), device = self._device, dtype=self.dtype) # floating base angular vel
            self.a = torch.zeros((cluster_size, 3), device = self._device, dtype=self.dtype) # floating base angular acc

    class JntStates:

        def __init__(self, 
                    n_dofs: int, 
                    cluster_size: int = 1, 
                    device: str = "cpu", 
                    dtype = torch.float32):

            self.dtype = dtype

            self._device = device

            self.q = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint positions
            self.v = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint velocities
            self.a = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations
            self.eff = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: str = "cpu", 
                dtype: torch.dtype = torch.float32):
        
        self.dtype = dtype

        self.backend = "torch" # forcing torch backend

        self.device = device

        if (self.backend != "torch"):

            self.device = "cpu"

        self.root_state = self.RootStates(cluster_size = cluster_size, 
                                        device = self.device, 
                                        dtype = self.dtype)

        self.jnt_state = self.JntStates(n_dofs = n_dofs, 
                                        cluster_size = cluster_size, 
                                        device = self.device, 
                                        dtype = self.dtype)
        
        self.n_dofs = n_dofs
        self.cluster_size = cluster_size

class RobotClusterCmd:

    class JntCmd:

        def __init__(self, 
                    n_dofs: int, 
                    cluster_size: int = 1, 
                    device: str = "cpu", 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device

            self.q = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint positions
            self.v = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint velocities
        #     self.a = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations
            self.eff = torch.zeros((cluster_size, n_dofs), device = self._device, dtype=self.dtype) # joint accelerations

    class RhcInfo:

        def __init__(self, 
                    cluster_size: int = 1, 
                    add_data_size: int = 1,
                    device: str = "cpu", 
                    dtype = torch.float32):
            
            self.dtype = dtype

            self._device = device
            
            self._add_data_size = add_data_size

            self.info =  torch.zeros((cluster_size, self._add_data_size), device = self._device, dtype=self.dtype)

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: str = "cpu", 
                dtype: torch.dtype = torch.float32):

        self.n_dofs = n_dofs
        self.cluster_size = cluster_size
    
        self.dtype = dtype

        self.backend = "torch" # forcing torch backend

        self.device = device

        if (self.backend != "torch"):

            self.device = "cpu"

        self.jnt_cmd = self.JntCmd(n_dofs = self.n_dofs, 
                                cluster_size = self.cluster_size, 
                                device = self.device, 
                                dtype = self.dtype)
        
        self.rhc_info = self.RhcInfo(cluster_size=self.cluster_size, 
                                device=self.device, 
                                dtype=self.dtype)



class Action(ABC):
        
        pass

ActionChild = TypeVar('ActionChild', bound='Action')