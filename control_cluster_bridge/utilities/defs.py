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
def aggregate_state_size(internal_jnt_number: int):

    # root p, q, v, omega + jnt q, v respectively
    aggregate_dim = 3 + \
            4 + \
            3 + \
            3 + \
            2 * internal_jnt_number # hardcoded

    return aggregate_dim

def aggregate_cmd_size(internal_jnt_number: int, 
                    aux_data_size: int):

    # jnt q, v, eff + aux data
    
    aggregate_dim = 3 * internal_jnt_number + \
            aux_data_size # hardcoded

    return aggregate_dim

def aggregate_refs_size(contact_n: int):

    # hardcoded
    phase_id = 1 + \
            contact_n 
            # + \
            # 1 + \
            # 1 + \
            # 2 + \
            # 1 + \
            # 2
                
    base_pose = 3 + 4
    com_pose = 3 + 4
    
    refs_size = phase_id + \
            base_pose + \
            com_pose

    return refs_size

def states_name():
    
    name = "RobotState"

    return name

def cmds_name():
    
    name = "RobotCmds"

    return name

def task_refs_name():
    
    name = "TaskRefs"

    return name

def trigger_flagname():

    name = "Trigger"

    return name

def reset_controllers_flagname():

    name = "ResetControllers"

    return name

def controllers_fail_flagname():

    name = "ControllersFail"

    return name

def launch_controllers_flagname():

    name = "LaunchControllers"

    return name

def launch_keybrd_cmds_flagname():

    name = "LaunchKeyboardCmds"

    return name

def env_selector_name():

    name = "EnvIndexSelector"

    return name

def solved_flagname():

    name = "Solved"

    return name

def cluster_size_name():
    
    name = "ClusterSize"

    return name 

def jnt_number_srvr_name():
    
    name = "JntNumberSrvr"

    return name

def jnt_names_client_name():

    name = "JntNamesClient"

    return name 

def jnt_number_client_name():

    name = "JntNumberClient"

    return name 

def additional_data_name():

    name = "AddData"

    return name

def n_contacts_name():

    name = "NContacts"

    return name

def shared_clients_count_name():

    name = "clients_counter"

    return name

def shared_sem_srvr_name():

    name = "semaphore_srvr"

    return name

def shared_srvr_nrows_name():

    name = "nrows_srvr"

    return name

def shared_srvr_ncols_name():

    name = "ncols_srvr"

    return name

def shared_sem_clients_count_name():

    name = "semaphore_clients_count"

    return name

class Journal:

    def __init__(self):

        self.warning = "warning"
        self.exception = "exception"
        self.info = "info"
        self.status = "status"