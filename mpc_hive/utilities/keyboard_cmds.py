# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of MPCHive and distributed under the General Public License version 2 license.
# 
# MPCHive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# MPCHive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MPCHive.  If not, see <http://www.gnu.org/licenses/>.
# 

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math

import numpy as np

class RefsFromKeyboard:

    def parse_contact_mapping(self, mapping_str):
        """
        Parses a string formatted like "0;1;3;2 " and extracts the numbers as a list of integers.
        Returns None if the input string is invalid.
        
        Args:
            mapping_str (str): The input string containing numbers separated by semicolons.
            
        Returns:
            List[int] or None: A list of integers extracted from the input string, or None if invalid.
        """
        try:
            # Strip any surrounding whitespace
            mapping_str = mapping_str.strip()
            
            # Split by semicolons and check for validity
            parts = mapping_str.split(';')
            if not all(part.isdigit() for part in parts if part):  # Ensure all parts are valid integers
                return None
            
            # Convert to integers and return as a list
            return [int(num) for num in parts if num]
        except Exception:
            # Catch any unexpected errors and return None
            return None

    def __init__(self, 
        namespace: str,
        shared_refs, 
        verbose = False,
        contact_mapping: str = "",
        env_idx: int = None):

        self._env_idx=env_idx

        self._contact_mapping=self.parse_contact_mapping(contact_mapping)
        if self._contact_mapping is None:
            self._contact_mapping=[0, 1, 2, 3] # key 7-9-1-3

        n_contacts=4
        
        self._verbose = verbose

        self.namespace = namespace

        self._shared_refs=shared_refs
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
        self._phase_id_current=0
        self._contact_dpos=0.02

        self._enable_flight_param_change=False
        self._d_flight_length=1
        self._d_flight_apex=0.01
        self._d_flight_end=0.01
        self._d_flength_enabled=False
        self._d_fapex_enabled=False
        self._d_fend_enabled=False
        self._d_fparam_enabled_contact_i = [False]*n_contacts

        self.enable_contact_pos_change= False
        self.enable_contact_pos_change_ci = [False]*n_contacts
        self.enable_contact_pos_change_xyz = [False]*3
        
        self.contact_pos_change_vals = np.zeros((3, n_contacts))
        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._init_shared_data()

    def _init_shared_data(self):

        self.env_index=None
        self.launch_keyboard_cmds=None
        if self._env_idx is None:
            # for detecting when commands are active
            self.launch_keyboard_cmds = SharedTWrapper(namespace = self.namespace,
                basename = "KeyboardCmdsLauncher",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Bool)
        
            self.launch_keyboard_cmds.run()

            # for sending to specific env
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
        
            self.env_index.run()
        
        self._init_ref_subscriber()

    def _init_ref_subscriber(self):
        
        if not self._shared_refs.is_running():
            self._shared_refs.run()
        
        self._contact_names=self._shared_refs.rob_refs.contact_pos.contact_names

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self._shared_refs is not None:

            self._shared_refs.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:
            if self.env_index is not None:
                self.env_index.synch_all(read=True, retry=True)
                env_index = self.env_index.get_numpy_mirror()
                self._env_idx=env_index[0, 0].item()
            self.cluster_idx = self._env_idx
            self.cluster_idx_np = self.cluster_idx
        
            self._shared_refs.rob_refs.synch_from_shared_mem()
            self._shared_refs.contact_flags.synch_all(read=True, retry=True)
            self._shared_refs.phase_id.synch_all(read=True, retry=True)
            self._shared_refs.flight_settings_req.synch_all(read=True, retry=True)
        
        else:
            
            self._shared_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.rob_refs.root_state.n_cols,
                                                read=False)
            self._shared_refs.rob_refs.contact_pos.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.rob_refs.contact_pos.n_cols,
                                                read=False)
            
            
            
            
            
            self._shared_refs.flight_settings_req.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.flight_settings_req.n_cols,
                                                read=False)
                                                
    def _update_base_height(self, 
                decrement = False):
        
        # update both base height and vel
        current_p_ref = self._shared_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np)
        if decrement:
            new_height_ref = current_p_ref[2] - self.height_dh
        else:
            new_height_ref = current_p_ref[2] + self.height_dh
        current_p_ref[2] = new_height_ref
        self._shared_refs.rob_refs.root_state.set(data_type="p",data=current_p_ref,
                                    robot_idxs=self.cluster_idx_np)
    
    def _update_navigation(self, 
                    type: str,
                    increment = True,
                    reset: bool=False):

        current_lin_v_ref = self._shared_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        current_omega_ref = self._shared_refs.rob_refs.root_state.get(data_type="omega", robot_idxs=self.cluster_idx_np)

        if not reset:
            if type=="frontal_lin" and not increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] - self.dxy
            if type=="frontal_lin" and increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] + self.dxy
            if type=="lateral_lin" and increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] - self.dxy
            if type=="lateral_lin" and not increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] + self.dxy
            if type=="vertical_lin" and not increment:
                # frontal motion
                current_lin_v_ref[2] = current_lin_v_ref[2] - self.dxy
            if type=="vertical_lin" and increment:
                # frontal motion
                current_lin_v_ref[2] = current_lin_v_ref[2] + self.dxy
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

        self._shared_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        self._shared_refs.rob_refs.root_state.set(data_type="omega",data=current_omega_ref,
                                    robot_idxs=self.cluster_idx_np)

    def _update_phase_id(self,
                phase_id: int = -1):

        phase_id_shared = self._shared_refs.phase_id.get_numpy_mirror()
        phase_id_shared[self.cluster_idx, :] = phase_id
        self._phase_id_current=phase_id

    def _update_flight_params(self, contact_idx: int, increment: bool = True):
        
        if self._d_flength_enabled:
            len_now=self._shared_refs.flight_settings_req.get(data_type="len_remain",
                    robot_idxs=self.cluster_idx,
                    contact_idx=contact_idx)
            if increment:
                len_now=len_now+self._d_flight_length
            else:
                len_now=len_now-self._d_flight_length
            self._shared_refs.flight_settings_req.set(data=np.array(len_now),
                    data_type="len_remain",
                    robot_idxs=self.cluster_idx,
                    contact_idx=contact_idx)
        
        if self._d_fapex_enabled:
            apex_now=self._shared_refs.flight_settings_req.get(data_type="apex_dpos",
                    robot_idxs=self.cluster_idx,
                    contact_idx=contact_idx)
            if increment:
                apex_now=apex_now+self._d_flight_apex
            else:
                apex_now=apex_now-self._d_flight_apex
            self._shared_refs.flight_settings_req.set(data=np.array(apex_now),
                data_type="apex_dpos",
                robot_idxs=self.cluster_idx,
                contact_idx=contact_idx)
        
        if self._d_fend_enabled:
            end_now=self._shared_refs.flight_settings_req.get(data_type="end_dpos",
                    robot_idxs=self.cluster_idx,
                    contact_idx=contact_idx)
            if increment:
                end_now=end_now+self._d_flight_end
            else:
                end_now=end_now-self._d_flight_end
            self._shared_refs.flight_settings_req.set(data=np.array(end_now),
                data_type="end_dpos",
                robot_idxs=self.cluster_idx,
                contact_idx=contact_idx)
       
    def _set_contacts(self,
                key,
                is_contact: bool = True):
        contact_flags = self._shared_refs.contact_flags.get_numpy_mirror()
        if key == "7":
            contact_flags[self.cluster_idx, self._contact_mapping[0]] = is_contact
        if key == "9":
            contact_flags[self.cluster_idx, self._contact_mapping[1]] = is_contact
        if key == "1":
            contact_flags[self.cluster_idx, self._contact_mapping[2]] = is_contact
        if key == "3":
            contact_flags[self.cluster_idx, self._contact_mapping[3]] = is_contact
    
    def _set_flight_params(self,
        key):

        if key=="F":
            self._enable_flight_param_change = not self._enable_flight_param_change
            info = f"Flight params change enabled: {self._enable_flight_param_change}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if key=="L":
            self._d_flength_enabled=not self._d_flength_enabled
            info = f"Flight length change enabled: {self._d_flength_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="A":
            self._d_fapex_enabled=not self._d_fapex_enabled
            info = f"Flight apex change enabled: {self._d_fapex_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="E":
            self._d_fend_enabled=not self._d_fend_enabled
            info = f"Flight end change enabled: {self._d_fend_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key=="o":
            self._d_fparam_enabled_contact_i[0]=not self._d_fparam_enabled_contact_i[0]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[0]} for contact 0"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="p":
            self._d_fparam_enabled_contact_i[1]=not self._d_fparam_enabled_contact_i[1]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[1]} for contact 1"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="k":
            self._d_fparam_enabled_contact_i[2]=not self._d_fparam_enabled_contact_i[2]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[2]} for contact 2"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="l":
            self._d_fparam_enabled_contact_i[3]=not self._d_fparam_enabled_contact_i[3]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[3]} for contact 3"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key=="+":
            if self._d_fparam_enabled_contact_i[0]:
                self._update_flight_params(contact_idx=0,
                    increment=True)
            if self._d_fparam_enabled_contact_i[1]:
                self._update_flight_params(contact_idx=1,
                    increment=True)
            if self._d_fparam_enabled_contact_i[2]:
                self._update_flight_params(contact_idx=2,
                    increment=True)
            if self._d_fparam_enabled_contact_i[3]:
                self._update_flight_params(contact_idx=3,
                    increment=True)
        if key=="-":
            if self._d_fparam_enabled_contact_i[0]:
                self._update_flight_params(contact_idx=0,
                    increment=False)
            if self._d_fparam_enabled_contact_i[1]:
                self._update_flight_params(contact_idx=1,
                    increment=False)
            if self._d_fparam_enabled_contact_i[2]:
                self._update_flight_params(contact_idx=2,
                    increment=False)
            if self._d_fparam_enabled_contact_i[3]:
                self._update_flight_params(contact_idx=3,
                    increment=False)
                
    def _set_phase_id(self,
                    key):

        if key == "p":
                    
            self.enable_phase_id_change = not self.enable_phase_id_change

            info = f"Phase ID change enabled: {self.enable_phase_id_change}"

            Journal.log(self.__class__.__name__,
                "_set_phase_id",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key == "0" and self.enable_phase_id_change:
            
            self._update_phase_id(phase_id = 0)

        elif key == "1" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 1)
        
        elif key == "2" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 2)
        
        elif key == "3" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 3)
        
        elif key == "4" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 4)
        
        elif key == "5" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 5)

        elif key == "6" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 6)

        elif key == "r" and self.enable_phase_id_change:
        
            self._update_phase_id(phase_id = -1)

    def _set_base_height(self,
                    key):

        if key == "h":
                    
            self.enable_heightchange = not self.enable_heightchange

            info = f"Base heightchange enabled: {self.enable_heightchange}"

            Journal.log(self.__class__.__name__,
                "_set_base_height",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        # if not self.enable_heightchange:
        #     self._update_base_height(reset=True)

        if key == "+" and self.enable_heightchange:
            self._update_base_height(decrement=False)
        
        if key == "-" and self.enable_heightchange:
            self._update_base_height(decrement=True)

    def _set_twist(self, 
                key):
        
        if key == "T":
            self.enable_twist = not self.enable_twist
            info = f"Twist change enabled: {self.enable_twist}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_twist:
            self._update_navigation(type="twist",reset=True)

        if self.enable_twist and key == "x":
            self.enable_twist_roll = not self.enable_twist_roll
            info = f"Twist roll change enabled: {self.enable_twist_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key == "y":
            self.enable_twist_pitch = not self.enable_twist_pitch
            info = f"Twist pitch change enabled: {self.enable_twist_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key == "z":
            self.enable_twist_yaw = not self.enable_twist_yaw
            info = f"Twist yaw change enabled: {self.enable_twist_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key == "+":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = True)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = True)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = True)
        if key == "-":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = False)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = False)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = False)
            
    def _set_linvel(self,
                key):

        if key == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"Navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(type="lin",reset=True)

        if key == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
        if key == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key == "+" and self.enable_navigation:
            self._update_navigation(type="vertical_lin", 
                            increment = True)
        if key == "-" and self.enable_navigation:
            self._update_navigation(type="vertical_lin",
                            increment = False)
    
    # def _set_contact_target_pos(self,
    #         key):

    #     if hasattr(key, 'char'):
            
    #         if key == "P":
    #             self.enable_contact_pos_change = not self.enable_contact_pos_change
    #             info = f"Contact pos change enabled: {self.enable_contact_pos_change}"
    #             Journal.log(self.__class__.__name__,
    #                 "_set_phase_id",
    #                 info,
    #                 LogType.INFO,
    #                 throw_when_excep = True)

    #             if not self.enable_contact_pos_change:
    #                 self.contact_pos_change_vals[:, :]=0
    #                 self.enable_contact_pos_change_xyz[0]=0
    #                 self.enable_contact_pos_change_xyz[1]=0
    #                 self.enable_contact_pos_change_xyz[2]=0
                
    #         if self.enable_contact_pos_change:
    #             if key == "x":
    #                 self.enable_contact_pos_change_xyz[0] = not self.enable_contact_pos_change_xyz[0]
    #             if key == "y":
    #                 self.enable_contact_pos_change_xyz[1] = not self.enable_contact_pos_change_xyz[1]
    #             if key == "z":
    #                 self.enable_contact_pos_change_xyz[2] = not self.enable_contact_pos_change_xyz[2]
        
    #             if key == "+":
    #                 self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
    #                     self.enable_contact_pos_change_ci)]+= self._contact_dpos
    #             if key == "-":
    #                 self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
    #                     self.enable_contact_pos_change_ci)]-= self._contact_dpos
                
    #             # not_enabled = [not x for x in self.enable_contact_pos_change_xyz]
    #             # self.contact_pos_change_vals[np.ix_(not_enabled,
    #             #         self.enable_contact_pos_change_ci)]= 0
                
    #     if key == Key.insert:
    #         self.enable_contact_pos_change_ci[0] = not self.enable_contact_pos_change_ci[0]
    #     if key == Key.page_up:
    #         self.enable_contact_pos_change_ci[1] = not self.enable_contact_pos_change_ci[1]
    #     if key == Key.delete:
    #         self.enable_contact_pos_change_ci[2] = not self.enable_contact_pos_change_ci[2]
    #     if key == Key.page_down:
    #         self.enable_contact_pos_change_ci[3] = not self.enable_contact_pos_change_ci[3]
                
    #     current_contact_pos_trgt = self._shared_refs.rob_refs.contact_pos.get(data_type="p", 
    #                                         robot_idxs=self.cluster_idx_np)
    #     current_contact_pos_trgt[:] = self.contact_pos_change_vals.flatten()

    #     self.contact_pos_change_vals

    def _on_press(self, key):

        cmds_active=True
        if self.launch_keyboard_cmds is not None:
            cmds_active=self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]
            
        if cmds_active:
        
            self._synch(read=True) # updates  data like
            # current cluster index

            if not self._read_from_stdin:
                if hasattr(key, 'char'):
                    key=key.char
                    # phase ids

            self._set_phase_id(key)
            # stepping phases (if phase id allows it)
            self._set_contacts(key=key, 
                        is_contact=False)
            # height change
            self._set_base_height(key)
            # (linear) navigation cmds
            self._set_linvel(key)
            # orientation (twist)
            self._set_twist(key)
            
            self._set_flight_params(key)

            # self._set_contact_target_pos(key)

            self._synch(read=False)

    def _on_release(self, key):
        cmds_active=True
        if self.launch_keyboard_cmds is not None:
            cmds_active=self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]
            
        if cmds_active:
            
            if not self._read_from_stdin:
                if hasattr(key, 'char'):
                    key=key.char
                    # phase ids

            self._set_contacts(key=key, 
                    is_contact=True)

            self._synch(read=False)

    def run(self, read_from_stdin: bool = False,
        release_timeout: float = 0.1):
        
        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True,type="lin")
        self._update_navigation(reset=True,type="twist")

        self._read_from_stdin=read_from_stdin
        if read_from_stdin:
            from mpc_hive.utilities.keyboard_listener_stdin import KeyListenerStdin
            import time

            with KeyListenerStdin(on_press=self._on_press, 
                on_release=self._on_release, 
                release_timeout=release_timeout) as listener:
                
                while not listener.done:
                    time.sleep(0.1)  # Keep the main thread alive
                listener.stop()

        else:
            from pynput import keyboard
            from pynput.keyboard import Key
        
            with keyboard.Listener(on_press=self._on_press, 
                        on_release=self._on_release) as listener:

                listener.join()

if __name__ == "__main__":  

    keyb_cmds = RefsFromKeyboard(namespace="kyon0", 
                            verbose=True,
                            env_idx=0)

    keyb_cmds.run()