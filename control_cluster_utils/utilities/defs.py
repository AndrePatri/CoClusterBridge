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
    phase_id = contact_n + 1
    base_pose = 3 + 4
    com_pos = 3
    
    refs_size = phase_id + \
            base_pose + \
            com_pos

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