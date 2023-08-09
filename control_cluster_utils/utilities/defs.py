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
        