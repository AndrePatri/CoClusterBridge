from control_cluster_utils.utilities.shared_mem import SharedMemClient

import torch

import time

import numpy as np

def profile_read_write_cmds_states():

    dtype = torch.float32
    dtype_np = np.float32

    n_reads = 100

    n_envs = 100
    n_jnts = 60

    clients_state = []

    for i in range(0, n_envs):

        print("Creating client n." + str(i))
        clients_state.append(SharedMemClient(n_envs, 
                                    n_jnts, 
                                    i, 
                                    'state', 
                                    dtype))
        
    clients_cmds = []

    for i in range(0, n_envs):

        print("Creating client n." + str(i))
        clients_cmds.append(SharedMemClient(n_envs, 
                                    n_jnts, 
                                    i, 
                                    'cmds', 
                                    dtype))
    
    
    a = np.zeros((clients_state[0].tensor_view.shape[0], \
                clients_state[0].tensor_view.shape[1]), dtype=dtype_np)
    
    for i in range(0, n_reads):

        print("state ###########################")

        for i in range(n_envs):
            
            t = time.monotonic()
            a = clients_state[i].tensor_view.numpy()
            print("time 2 read state: " + str(time.monotonic() - t ))
            print("idx: " + str(i))

        print("cmds ###########################")

        for i in range(n_envs):
            
            t = time.monotonic()
            clients_cmds[i].tensor_view[:, :] = torch.from_numpy(a)
            print("time 2 assign cmds: " + str(time.monotonic() - t ))
            print("idx: " + str(i))

        time.sleep(0.1)

def profile_writing_bool_array():

    n_samples = 100

    n_envs = 100
    n_jnts = 1

    dtype = torch.bool
    clients = []

    for i in range(0, n_envs):

        clients.append(SharedMemClient(n_envs, n_jnts, i, 
                        "solved", 
                        dtype=dtype))

    for i in range(0, n_samples):
        
        print("###########################")

        for i in range(0, n_envs):

            t = time.monotonic()
            
            # for i in range(server.tensor_view.shape[0]):

            clients[i].set_bool(True)
            # clients[i].tensor_view[0, 0] = True
            print(f"{i} -> time 2 write bool array elements: " + str(time.monotonic() - t ))

            print(clients[i].tensor_view)

        time.sleep(0.1)

if __name__ == "__main__":
    
    profile_writing_bool_array()

    profile_read_write_cmds_states()
        

        
