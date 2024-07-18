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
import numpy as np

import xml.etree.ElementTree as ET

from typing import List
from SharsorIPCpp.PySharsorIPC import Journal, LogType

class RobotHomer:

    def __init__(self, 
            srdf_path: str, 
            jnt_names_prb: List[str] = None,
            filter: bool = True):

        self._filter=filter

        self.srdf_path = srdf_path

        self.jnt_names_prb = jnt_names_prb # coming from controller
        
        # open srdf and parse the homing field
        
        with open(srdf_path, 'r') as file:
            self._srdf_content = file.read()

        try:
            self._srdf_root = ET.fromstring(self._srdf_content)
        except ET.ParseError as e:
            exception = f"could not read SRDF properly!!"
            Journal.log(self.__class__.__name__,
                        "__init__",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
            
        # Find all the 'joint' elements within 'group_state' with the name attribute and their values
        srdf_joints = self._srdf_root.findall(".//group_state[@name='home']/joint")

        self._homing_value_map = {}

        self.jnt_names_srdf = []
        self.homing_srdf = []
        for srdf_joint in srdf_joints:
            joint_name = srdf_joint.attrib['name']
            joint_value = srdf_joint.attrib['value']
            self.jnt_names_srdf.append(joint_name)
            self.homing_srdf.append(joint_value)
            self._homing_value_map[joint_name] =  float(joint_value) # joint name -> homing value

        if self.jnt_names_prb is None:
            # we use the same joints in the SRDF
            self.jnt_names_prb = self.jnt_names_srdf

        if self._filter: # remove some joints
            self.jnt_names_prb = self._filter_jnt_names(self.jnt_names_prb)

        self.n_dofs_prb = len(self.jnt_names_prb)
        self.n_dofs_srdf = len(self.jnt_names_srdf)
        if not self.n_dofs_prb==self.n_dofs_srdf:
            warn = f"Found {self.n_dofs_srdf} jnt in SRDF, while provided ones are {self.n_dofs_prb}!"
            Journal.log(self.__class__.__name__,
                        "__init__",
                        warn,
                        LogType.WARN)
        self._check_jnt_names()

        additional_jnts=list(set(self.jnt_names_srdf)-set(self.jnt_names_prb))
        self._homing_value_map_prb=self._remove_keys_from_dict(self._homing_value_map,additional_jnts)

        self.joint_idx_map_prb = {}
        for joint in range(0, self.n_dofs_prb): # go through joints in the order they were provided
            self.joint_idx_map_prb[self.jnt_names_prb[joint]] = joint # jnt name in prb -> joint index in homing matrix

        self._homing = np.full((1, self.n_dofs_prb), 
                        0.0, 
                        dtype=np.float32) # homing configuration
        
        self._assign_homing()

    def _assign_homing(self):
        # assign homing prb
        for joint in self.jnt_names_prb: # joint is guaranteed to be in _homing_value_map (check was performed)
            self._homing[:, self.joint_idx_map_prb[joint]] = self.joint_idx_map_prb[joint]

    def get_homing(self):
        return self._homing.flatten()
    
    def get_homing_vals(self,jnt_names:List[str]):
        homing_list=[]
        for jnt_name in jnt_names: # using srdf map, since it may contain more joints
            homing_list.append(self._homing_value_map[jnt_name])
        return homing_list
    
    def get_homing_map(self,from_prb:bool=True):
        if from_prb:
            return self._homing_value_map_prb
        else:
            return self._homing_value_map
    
    def _remove_keys_from_dict(self,dictionary, keys_to_remove):
        return {key: value for key, value in dictionary.items() if key not in keys_to_remove}

    def _check_jnt_names(self):
        # Convert both lists to sets for efficient membership checking
        names_prb = set(self.jnt_names_prb)
        names_srdf = set(self.jnt_names_srdf)
        
        # Check if all elements of set1 are present in set2
        ok=names_prb.issubset(names_srdf)
        if not ok:
            excepion = f"Some of the provided joint names are not present in the SRDF!"
            Journal.log(self.__class__.__name__,
                        "_check_jnt_names",
                        excepion,
                        throw_when_excep = True)

    def _filter_jnt_names(self, 
                        names: List[str]):

        to_be_removed = ["universe", 
                        "reference", 
                        "world", 
                        "floating", 
                        "floating_base"]
        
        for name in to_be_removed:

            if name in names:
                names.remove(name)

        return names
    