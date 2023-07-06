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
                    device: str = "cpu"):

            self._device = device

            self.q = torch.zeros((cluster_size, 4), device = self._device) # floating base orientation (quaternion)
            self.v = torch.zeros((cluster_size , 3), device = self._device) # floating base angular vel
            self.a = torch.zeros((cluster_size, 3), device = self._device) # floating base angular acc

    class JntStates:

        def __init__(self, 
                    n_dofs: int, 
                    cluster_size: int = 1, 
                    device: str = "cpu"):

            self._device = device

            self.q = torch.zeros((cluster_size, n_dofs), device = self._device) # joint positions
            self.v = torch.zeros((cluster_size, n_dofs), device = self._device) # joint velocities
            self.a = torch.zeros((cluster_size, n_dofs), device = self._device) # joint accelerations

    def __init__(self, 
                n_dofs: int, 
                cluster_size: int = 1, 
                backend: str = "torch", 
                device: str = "cpu"):

        self.backend = "torch" # forcing torch backend

        self.device = device

        if (self.backend != "torch"):

            self.device = "cpu"

        self.root_state = self.RootStates(cluster_size = cluster_size, 
                                        device = self.device)

        self.jnt_state = self.JntStates(n_dofs = n_dofs, 
                                        cluster_size = cluster_size, 
                                        device = self.device)
        
        self.n_dofs = n_dofs
        self.cluster_size = cluster_size

class Action(ABC):
        
        pass

ActionChild = TypeVar('ActionChild', bound='Action')