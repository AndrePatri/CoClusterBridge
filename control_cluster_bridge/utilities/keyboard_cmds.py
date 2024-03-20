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
import torch

from pynput import keyboard

from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import dtype

from control_cluster_bridge.utilities.math_utils import incremental_rotate

import math

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
        self.dxy = 0.05 # [m]
        self.dtheta_z = 1.0 * math.pi / 180.0 # [rad]

        self.enable_phase_id_change = False

        self.rhc_refs = None

        self.cluster_idx = -1
        self.cluster_idx_torch = torch.tensor(self.cluster_idx)

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
                                with_gpu_mirror=False, 
                                with_torch_view=False,
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

            self.cluster_idx = self.env_index.get_numpy_view()[0, 0].item()
            self.cluster_idx_torch = self.cluster_idx
        
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
        
        current_p_ref = self.rhc_refs.rob_refs.root_state.get_p(robot_idxs=self.cluster_idx_torch)

        if decrement:

            new_height_ref = current_p_ref[0, 2] - self.height_dh

        else:

            new_height_ref = current_p_ref[0, 2] + self.height_dh

        current_p_ref[0, 2] = new_height_ref

        self.rhc_refs.rob_refs.root_state.set_p(p = current_p_ref,
                                    robot_idxs=self.cluster_idx_torch)
    
    def _update_navigation(self, 
                    lateral = None, 
                    orientation = None,
                    increment = True):

        current_p_ref = self.rhc_refs.rob_refs.root_state.get_p(robot_idxs=self.cluster_idx_torch)

        current_q_ref = self.rhc_refs.rob_refs.root_state.get_q(robot_idxs=self.cluster_idx_torch)

        if lateral is not None and lateral and increment:
            # lateral motion
            
            current_p_ref[0, 1] = current_p_ref[0, 1] - self.dxy

        if lateral is not None and lateral and not increment:
            # lateral motion
            
            current_p_ref[0, 1] = current_p_ref[0, 1] + self.dxy

        if lateral is not None and not lateral and not increment:
            # frontal motion
            
            current_p_ref[0, 0] = current_p_ref[0, 0] - self.dxy

        if lateral is not None and not lateral and increment:
            # frontal motion
            
            current_p_ref[0, 0] = current_p_ref[0, 0] + self.dxy

        if orientation is not None and orientation and increment:
            
            # rotate counter-clockwise
            current_q_ref[0, :] = incremental_rotate(current_q_ref.flatten(), 
                            self.dtheta_z, 
                            [0, 0, 1])

        if orientation is not None and orientation and not increment:
            
            # rotate counter-clockwise
            current_q_ref[0, :] = incremental_rotate(current_q_ref.flatten(), 
                            -self.dtheta_z, 
                            [0, 0, 1])

        self.rhc_refs.rob_refs.root_state.set_p(p = current_p_ref,
                                    robot_idxs=self.cluster_idx_torch)
        
        self.rhc_refs.rob_refs.root_state.set_q(q = current_q_ref,
                                    robot_idxs=self.cluster_idx_torch)

    def _update_phase_id(self,
                phase_id: int = -1):

        self.rhc_refs.phase_id.get_numpy_view()[self.cluster_idx, 0] = phase_id

    def _set_contacts(self,
                key,
                is_contact: bool = True):
        
        if key.char == "7":
                    
            self.rhc_refs.contact_flags.get_numpy_view()[self.cluster_idx, 0] = is_contact

        if key.char == "9":
            
            self.rhc_refs.contact_flags.get_numpy_view()[self.cluster_idx, 1] = is_contact

        if key.char == "1":
            
            self.rhc_refs.contact_flags.get_numpy_view()[self.cluster_idx, 2] = is_contact

        if key.char == "3":
            
            self.rhc_refs.contact_flags.get_numpy_view()[self.cluster_idx, 3] = is_contact
    
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
        
        if key.char == "6" and self.enable_navigation:
            
            self._update_navigation(lateral = True, 
                            increment = True)

        if key.char == "4" and self.enable_navigation:
            
            self._update_navigation(lateral = True, 
                            increment = False)
        
        if key.char == "8" and self.enable_navigation:
            
            self._update_navigation(lateral = False, 
                            increment = True)
        
        if key.char == "2" and self.enable_navigation:
            
            self._update_navigation(lateral = False, 
                            increment = False)
        
        if key == keyboard.Key.left and self.enable_navigation:
                
            self._update_navigation(orientation=True,
                                increment = True)

        if key == keyboard.Key.right and self.enable_navigation:
            
            self._update_navigation(orientation=True,
                                increment = False)
                
    def _on_press(self, key):

        if self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]:
            
            self._synch(read=True) # updates  data like
            # current cluster index

            if hasattr(key, 'char'):
                
                # print('Key {0} pressed.'.format(key.char))
                
                # phase ids
                self._set_phase_id(key)

                # stepping phases (if phase id allows it)
                self._set_contacts(key=key, 
                            is_contact=False)
                
                # height change
                self._set_base_height(key)

                # navigation
                self._set_navigation(key)

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
        
        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = RhcRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.run()