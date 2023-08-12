from control_cluster_utils.utilities.rhc_defs import RobotStateChild, RobotCmds, RobotState
from control_cluster_utils.utilities.control_cluster_defs import RobotClusterState, RobotClusterCmd

import torch

if __name__ == "__main__":

    cluster_size = 3
    ndofs = 4
    add_info_size = 2

    cluster_cmds = RobotClusterCmd(n_dofs=ndofs, 
                        cluster_size=cluster_size, 
                        add_data_size=add_info_size)
    
    q_cmd_srvr = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=cluster_cmds.dtype)
    v_cmd_srvr = torch.add(q_cmd_srvr, 1)
    eff_cmd_srvr = torch.add(v_cmd_srvr, 1)
    info_srvr =torch.tensor([[123213.0, 2342342.0]], dtype=cluster_cmds.dtype)

    server_jnt_names = ["joint_puzzo", "joint_gnegne£$", "hai_rotto_il_ca**o", "scibijoint0978"]

    client_jnt_names = ["joint_puzzo", "hai_rotto_il_ca**o", "joint_gnegne£$", "scibijoint0978"]
    expected_q_client_side = torch.tensor([[1.0, 3.0, 2.0, 4.0]], dtype=cluster_cmds.dtype)
    to_client = [server_jnt_names.index(element) for element in client_jnt_names]
    
    print("Initial client q:")
    print(cluster_cmds.jnt_cmd.q)

    srv_cmds = []

    for i in range(0, cluster_size):
        
        cmd = RobotCmds(n_dofs=ndofs, 
                    cluster_size=cluster_size, 
                    index=i, 
                    verbose=True, 
                    jnt_remapping=to_client, 
                    add_info_size=add_info_size)
                                  
        cmd.jnt_cmd.set_q(torch.add(q_cmd_srvr, i))

        cmd.jnt_cmd.set_v(torch.add(v_cmd_srvr, i))

        cmd.jnt_cmd.set_eff(torch.add(eff_cmd_srvr, i))

        cmd.slvr_state.set_info(torch.add(info_srvr, i))

        srv_cmds.append(cmd)
    
    cluster_cmds.synch()
    
    print("Expected row client side: \n" + str(expected_q_client_side))
    print("Client side q: \n" + str(cluster_cmds.jnt_cmd.q))
    print("Client side v: \n" + str(cluster_cmds.jnt_cmd.v))
    print("Client side eff: \n" + str(cluster_cmds.jnt_cmd.eff))
    print("Client side info: \n" + str(cluster_cmds.rhc_info.info))

    print("Client side tensor view: \n" + str(cluster_cmds.shared_memman.tensor_view[:, :]))

    # for i in range(0, cluster_size):

    #     srv_cmds[i].terminate()