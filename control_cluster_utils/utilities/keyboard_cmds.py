# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of ControlClusterUtils and distributed under the General Public License version 2 license.
# 
# ControlClusterUtils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# ControlClusterUtils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with ControlClusterUtils.  If not, see <http://www.gnu.org/licenses/>.
# 
import torch

from pynput import keyboard

from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs
from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.shared_mem import SharedMemSrvr
from control_cluster_utils.utilities.defs import cluster_size_name
from control_cluster_utils.utilities.defs import n_contacts_name
from control_cluster_utils.utilities.defs import launch_keybrd_cmds_flagname
from control_cluster_utils.utilities.defs import env_selector_name

from control_cluster_utils.utilities.math_utils import incremental_rotate

import math

class RhcRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._terminated = False

        self.contacts = None 
        
        self.enable_heightchange = False
        self.com_height_dh = 0.008 # [m]

        self.enable_navigation = False
        self.dxy = 0.05 # [m]
        self.dtheta_z = 1.0 * math.pi / 180.0 # [rad]

        self.cluster_size = -1
        self.n_contacts = -1

        self.cluster_size_clnt = None

        self.rhc_task_refs = []

        self._init_shared_data()

    def _init_shared_data(self):

        wait_amount = 0.05
        
        self.cluster_size_clnt = SharedMemClient(name=cluster_size_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self._verbose)
        self.cluster_size_clnt.attach()
        
        self.n_contacts_clnt = SharedMemClient(name=n_contacts_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self._verbose)
        self.n_contacts_clnt.attach()

        self.launch_keyboard_cmds = SharedMemSrvr(n_rows=1, 
                                        n_cols=1, 
                                        name=launch_keybrd_cmds_flagname(), 
                                        namespace=self.namespace,
                                        dtype=torch.bool)
        self.launch_keyboard_cmds.start()
        self.launch_keyboard_cmds.reset_bool(False)

        self.env_index = SharedMemClient(name=env_selector_name(), 
                                        namespace=self.namespace, 
                                        dtype=torch.int64, 
                                        wait_amount=wait_amount, 
                                        verbose=self._verbose)
        self.env_index.attach()
        self.env_index.tensor_view[0, 0] = 0 # inizialize to 1st environment

        # while self.launch_keyboard_cmds.get_clients_count() != 1:

        #     print("[RhcRefsFromKeyboard]" + self._init_shared_data.__name__ + ":"\
        #         " waiting for exactly one client to connect")

        self.cluster_size = self.cluster_size_clnt.tensor_view[0, 0].item()
        self.n_contacts = self.n_contacts_clnt.tensor_view[0, 0].item()
        
        # self.cluster_idx = self.env_index.tensor_view[0, 0]
        
        self.contacts = torch.tensor([[True] * self.n_contacts], 
                        dtype=torch.float32)
        
        self._init_shared_task_data()

    def _init_shared_task_data(self):

        # view of rhc references
        for i in range(0, self.cluster_size):

            self.rhc_task_refs.append(RhcTaskRefs( 
                n_contacts=self.n_contacts,
                index=i,
                q_remapping=None,
                namespace=self.namespace,
                dtype=torch.float32, 
                verbose=self._verbose))
        
    def __del__(self):

        if not self._terminated:

            self._terminate()
    
    def _terminate(self):
        
        self._terminated = True

        self.__del__()
    
    def _update(self):

        self.cluster_idx = self.env_index.tensor_view[0, 0].item()

    def _set_cmds(self):

        self.rhc_task_refs[self.cluster_idx].phase_id.set_contacts(
                                self.contacts)

    def _update_com_height(self, 
                decrement = False):
        
        current_com_ref = self.rhc_task_refs[self.cluster_idx].com_pose.get_com_height()

        if decrement:

            new_com_ref = current_com_ref - self.com_height_dh

        else:

            new_com_ref = current_com_ref + self.com_height_dh

        self.rhc_task_refs[self.cluster_idx].com_pose.set_com_height(new_com_ref)
    
    def _update_navigation(self, 
                    lateral = None, 
                    orientation = None,
                    increment = True):

        current_com_pos_ref = self.rhc_task_refs[self.cluster_idx].com_pose.get_com_pos()
        current_q_ref = self.rhc_task_refs[self.cluster_idx].base_pose.get_q()

        if lateral is not None and lateral and increment:
            # lateral motion
            
            current_com_pos_ref[:, 1] = current_com_pos_ref[:, 1] + self.dxy

        if lateral is not None and lateral and not increment:
            # lateral motion
            
            current_com_pos_ref[:, 1] = current_com_pos_ref[:, 1] - self.dxy

        if lateral is not None and not lateral and not increment:
            # frontal motion
            
            current_com_pos_ref[:, 0] = current_com_pos_ref[:, 0] - self.dxy

        if lateral is not None and not lateral and increment:
            # frontal motion
            
            current_com_pos_ref[:, 0] = current_com_pos_ref[:, 0] + self.dxy

        if orientation is not None and orientation and increment:
            
            q_result = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
            # rotate counter-clockwise
            q_result[0] = incremental_rotate(current_q_ref[0], 
                            self.dtheta_z, 
                            [0, 0, 1])
            current_q_ref = q_result

        if orientation is not None and orientation and not increment:
            
            # rotate clockwise
            # rotate counter-clockwise
            q_result = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
            q_result[0] = incremental_rotate(current_q_ref[0], 
                            - self.dtheta_z, 
                            [0, 0, 1])
            current_q_ref = q_result

        self.rhc_task_refs[self.cluster_idx].com_pose.set_com_pos(current_com_pos_ref)
        # self.rhc_task_refs[self.cluster_idx].com_pose.set_com_q(current_com_q_ref)
        self.rhc_task_refs[self.cluster_idx].base_pose.set_q(current_q_ref)

    def _on_press(self, key):

        if self.launch_keyboard_cmds.all():
            
            self._update() # updates  data like
            # current cluster index

            if hasattr(key, 'char'):
                
                print('Key {0} pressed.'.format(key.char))
                
                # stepping ph
                if key.char == "7":
                    
                    self.contacts[0, 0] = False

                if key.char == "9":
                    
                    self.contacts[0, 1] = False

                if key.char == "1":
                    
                    self.contacts[0, 2] = False

                if key.char == "3":
                    
                    self.contacts[0, 3] = False
                
                # height change
                if key.char == "h" and not self.enable_heightchange:
                    
                    self.enable_heightchange = True

                if key.char == "+" and self.enable_heightchange:

                    self._update_com_height(decrement=False)
                
                if key.char == "-" and self.enable_heightchange:

                    self._update_com_height(decrement=True)

                # navigation
                if key.char == "n" and not self.enable_navigation:
                    
                    self.enable_navigation = True

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

            self._set_cmds()

    def _on_release(self, key):

        if self.launch_keyboard_cmds.all():
            
            self._update() # updates  data like
            # current cluster index
            
            print(key)
            if hasattr(key, 'char'):
                
                # print('Key {0} released.'.format(key.char))

                if key.char == "7":
                    
                    self.contacts[0, 0] = True

                if key.char == "9":
                    
                    self.contacts[0, 1] = True

                if key.char == "1":
                    
                    self.contacts[0, 2] = True

                if key.char == "3":
                    
                    self.contacts[0, 3] = True

                if key == keyboard.Key.esc:

                    self._terminate()  # Stop listener

            self._set_cmds()

    def start(self):

        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = RhcRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.start()