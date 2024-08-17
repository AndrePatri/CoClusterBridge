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
from pynput import keyboard

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import dtype

import math

import numpy as np

class RhcRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._closed = False
        
        self.enable_heightchange = False
        self.height_dh = 0.02 # [m]

        self.enable_navigation = False
        self.enable_twist = False
        self.enable_twist_roll = False
        self.enable_twist_pitch = False
        self.enable_twist_yaw = False

        self.dxy = 0.05 # [m]
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self.enable_phase_id_change = False

        self.rhc_refs = None

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._init_shared_data()

    def _init_shared_data(self):

        self.launch_keyboard_cmds = SharedTWrapper(namespace = self.namespace,
                basename = "KeyboardCmdsLauncher",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Bool)
        
        self.launch_keyboard_cmds.run()

        self.env_index = SharedTWrapper(namespace = self.namespace,
                basename = "EnvSelector",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Int)
        
        self.env_index.run()
        
        self._init_rhc_ref_subscriber()

    def _init_rhc_ref_subscriber(self):

        self.rhc_refs = RhcRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=False, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2)

        self.rhc_refs.run()

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self.rhc_refs is not None:

            self.rhc_refs.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:

            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self.cluster_idx = env_index[0, 0].item()
            self.cluster_idx_np = self.cluster_idx
        
            self.rhc_refs.rob_refs.synch_from_shared_mem()
            self.rhc_refs.contact_flags.synch_all(read=True, retry=True)
            self.rhc_refs.phase_id.synch_all(read=True, retry=True)
        
        else:
            
            self.rhc_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self.rhc_refs.rob_refs.root_state.n_cols,
                                                read=False)

            self.rhc_refs.contact_flags.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self.rhc_refs.contact_flags.n_cols,
                                                read=False)
            
            self.rhc_refs.phase_id.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self.rhc_refs.phase_id.n_cols,
                                                read=False)
                                                
    def _update_base_height(self, 
                decrement = False):
        
        # current_p_ref = self.rhc_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np)
        # if decrement:
        #     new_height_ref = current_p_ref[2] - self.height_dh
        # else:
        #     new_height_ref = current_p_ref[2] + self.height_dh
        # current_p_ref[2] = new_height_ref
        # self.rhc_refs.rob_refs.root_state.set(data_type="p",data=current_p_ref,
        #                             robot_idxs=self.cluster_idx_np)
        
        current_lin_v_ref = self.rhc_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        if decrement:
            new_ref = current_lin_v_ref[2] - self.dxy
        else:
            new_ref = current_lin_v_ref[2] + self.dxy
        current_lin_v_ref[2] = new_ref
        self.rhc_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        
    def _update_navigation(self, 
                    type: str,
                    increment = True,
                    reset: bool=False):

        current_lin_v_ref = self.rhc_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        current_omega_ref = self.rhc_refs.rob_refs.root_state.get(data_type="omega", robot_idxs=self.cluster_idx_np)

        if not reset:
            if type=="lateral_lin" and increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] - self.dxy
            if type=="lateral_lin" and not increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] + self.dxy
            if type=="frontal_lin" and not increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] - self.dxy
            if type=="frontal_lin" and increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] + self.dxy
            if type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_omega_ref[0] = current_omega_ref[0] + self._dtwist 
            if type=="twist_roll" and not increment:
                current_omega_ref[0] = current_omega_ref[0] - self._dtwist 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_omega_ref[1] = current_omega_ref[1] + self._dtwist 
            if type=="twist_pitch" and not increment:
                current_omega_ref[1] = current_omega_ref[1] - self._dtwist 
            if type=="twist_yaw" and increment:
                # rotate counter-clockwise
                current_omega_ref[2] = current_omega_ref[2] + self._dtwist 
            if type=="twist_yaw" and not increment:
                current_omega_ref[2] = current_omega_ref[2] - self._dtwist 
        else:
            if "twist" in type:
                current_omega_ref[:]=0
            if "lin" in type:
                current_lin_v_ref[:]=0

        self.rhc_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        self.rhc_refs.rob_refs.root_state.set(data_type="omega",data=current_omega_ref,
                                    robot_idxs=self.cluster_idx_np)

    def _update_phase_id(self,
                phase_id: int = -1):

        phase_id = self.rhc_refs.phase_id.get_numpy_mirror()
        phase_id[self.cluster_idx, 111] = phase_id

    def _set_contacts(self,
                key,
                is_contact: bool = True):
        contact_flags = self.rhc_refs.contact_flags.get_numpy_mirror()
        if key.char == "7":
            contact_flags[self.cluster_idx, 0] = is_contact
        if key.char == "9":
            contact_flags[self.cluster_idx, 1] = is_contact
        if key.char == "1":
            contact_flags[self.cluster_idx, 2] = is_contact
        if key.char == "3":
            contact_flags[self.cluster_idx, 3] = is_contact
    
    def _set_phase_id(self,
                    key):

        if key.char == "p":
                    
            self.enable_phase_id_change = not self.enable_phase_id_change

            info = f"Phase ID change enabled: {self.enable_phase_id_change}"

            Journal.log(self.__class__.__name__,
                "_set_phase_id",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key.char == "0" and self.enable_phase_id_change:
            
            self._update_phase_id(phase_id = 0)

        elif key.char == "1" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 1)
        
        elif key.char == "2" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 2)
        
        elif key.char == "3" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 3)
        
        elif key.char == "4" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 4)
        
        elif key.char == "5" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 5)

        elif key.char == "6" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 6)

        elif key.char == "r" and self.enable_phase_id_change:
        
            self._update_phase_id(phase_id = -1)

    def _set_base_height(self,
                    key):

        if key.char == "h":
                    
            self.enable_heightchange = not self.enable_heightchange

            info = f"Base heightchange enabled: {self.enable_heightchange}"

            Journal.log(self.__class__.__name__,
                "_set_base_height",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key.char == "+" and self.enable_heightchange:
            self._update_base_height(decrement=False)
        
        if key.char == "-" and self.enable_heightchange:
            self._update_base_height(decrement=True)

    def _set_twist(self, 
                key):
        
        if key.char == "T":
            self.enable_twist = not self.enable_twist
            info = f"Twist change enabled: {self.enable_twist}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_twist:
            self._update_navigation(type="twist",reset=True)

        if self.enable_twist and key.char == "x":
            self.enable_twist_roll = not self.enable_twist_roll
            info = f"Twist roll change enabled: {self.enable_twist_roll}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "y":
            self.enable_twist_pitch = not self.enable_twist_pitch
            info = f"Twist pitch change enabled: {self.enable_twist_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "z":
            self.enable_twist_yaw = not self.enable_twist_yaw
            info = f"Twist yaw change enabled: {self.enable_twist_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key.char == "+":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = True)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = True)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = True)
        if key.char == "-":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = False)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = False)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = False)
            
    def _set_navigation(self,
                key):

        if key.char == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"Navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(type="lin",reset=True)

        if key.char == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key.char == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key.char == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key.char == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
                
    def _on_press(self, key):

        if self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]:
            
            self._synch(read=True) # updates  data like
            # current cluster index

            if hasattr(key, 'char'):
                                
                # phase ids
                # self._set_phase_id(key)
                # stepping phases (if phase id allows it)
                self._set_contacts(key=key, 
                            is_contact=False)
                # height change
                self._set_base_height(key)
                # (linear) navigation cmds
                self._set_navigation(key)
                # orientation (twist)
                self._set_twist(key)

            self._synch(read=False)

    def _on_release(self, key):

        if self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]:
            
            if hasattr(key, 'char'):
                
                # print('Key {0} released.'.format(key.char))

                self._set_contacts(key=key, 
                            is_contact=True)

                if key == keyboard.Key.esc:

                    self._close()

            self._synch(read=False)

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True,type="lin")
        self._update_navigation(reset=True,type="twist")

        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = RhcRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.run()