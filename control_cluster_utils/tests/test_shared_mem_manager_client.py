from control_cluster_utils.utilities.shared_mem import SharedMemClient

import torch

import time

import random 

if __name__ == "__main__":
    
    n_envs = 100
    n_jnts = 40

    clients = []

    for i in range(0, n_envs):

        print("Creating client n." + str(i))
        clients.append(SharedMemClient(n_envs, 
                                    n_jnts, 
                                    i, 
                                    'state', 
                                    torch.float32))
        
    while True:

        print("###########################")

        for i in range(n_envs):
            
            # t = time.monotonic()
            # clients[i].tensor_view[:, :] = torch.full((1, clients[i].tensor_view.shape[1]), 
            #                             i + random.random(),
            #                             dtype=clients[i].dtype)
            
            # print("time 2 copy: " + str(time.monotonic() - t ))
            # print("idx: " + str(i))
            print(clients[i].tensor_view)

        time.sleep(1.0)

        
