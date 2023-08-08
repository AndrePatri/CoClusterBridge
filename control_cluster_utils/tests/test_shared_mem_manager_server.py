from control_cluster_utils.utilities.shared_mem import SharedMemSrvr

import torch

import time

import random

if __name__ == "__main__":

    n_envs = 1
    n_jnts = 40

    server = SharedMemSrvr(n_envs, n_jnts, "state", 
                    dtype=torch.float32)
    for i in range(server.tensor_view.shape[0]):

        server.tensor_view[i, :] = torch.full((1, server.tensor_view.shape[1]), 
                                        i,
                                        dtype=server.dtype)
        
    
    while True:
        
        print("###########################")
        
        t = time.monotonic()
        server.tensor_view[:, :] = torch.full((server.tensor_view.shape[0], server.tensor_view.shape[1]), 
                                    random.random(),
                                    dtype=server.dtype)
        
        print("time 2 copy: " + str(time.monotonic() - t ))
        print(server.tensor_view)
        time.sleep(1.0)