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

def states_name():
    
    name = "RobotState"

    return name

def cmds_name():
    
    name = "RobotCmds"

    return name

def trigger_flagname():

    name = "Trigger"

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

    name = "JntNames"

    return name 

def jnt_number_client_name():

    name = "JntNumberClient"

    return name 

def additional_data_name():

    name = "AddData"

    return name

def srvr_writing_name():

    name = "SrvrWriting"

    return name

def client_writing_name():

    name = "ClientWriting"

    return name