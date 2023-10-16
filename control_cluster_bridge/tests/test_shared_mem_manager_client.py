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
from control_cluster_bridge.utilities.shared_mem import SharedMemClient

import torch

import time

import numpy as np

def profile_read_write_cmds_states():

    dtype = torch.float32
    dtype_np = np.float32

    n_reads = 10000

    n_envs = 100
    n_jnts = 60

    clients_state = []

    client = SharedMemClient(name = 'state', 
                            client_index=0, 
                            dtype=dtype)
    client.attach()
            
    # a = np.zeros((clients_state[0].tensor_view.shape[0], \
    #             clients_state[0].tensor_view.shape[1]), dtype=dtype_np)
    
    for i in range(0, n_reads):

        print("state ###########################")
        
        t = time.perf_counter()
        a = client.tensor_view.clone()
        t_end = time.perf_counter() - t 
        print("time to read state (with cloning): " + str(t_end))
        print("tensor: " + str(a))

        # print("cmds ###########################")

        # for i in range(n_envs):
            
        #     t = time.perf_counter()
        #     clients_cmds[i].tensor_view[:, :] = torch.from_numpy(a)
        #     t_end = time.perf_counter() - t 
        #     print("time 2 assign cmds: " + str(t_end))
        #     print("idx: " + str(i))

        time.sleep(0.1)

def profile_writing_bool_array():

    n_samples = 10

    n_envs = 100
    n_jnts = 1

    dtype = torch.bool
    clients = []

    for i in range(0, n_envs):

        clients.append(SharedMemClient("solved", 
                        i,
                        dtype=dtype))
        clients[i].attach()

    for i in range(0, n_samples):
        
        print("###########################")

        for j in range(0, n_envs):
            
            # for i in range(server.tensor_view.shape[0]):

            if i > n_samples/2:
                t = time.perf_counter()
                clients[j].set_bool(True)
                # clients[i].tensor_view[0, 0] = True
                t_end = time.perf_counter() - t 
                print(f"{j} -> time 2 write bool array elements: " + str(t_end))

            print(clients[i].tensor_view)

        time.sleep(0.1)

def profile_reading_global_bool():

    n_samples = 100

    n_envs = 100

    dtype = torch.bool
    clients = []

    for i in range(0, n_envs):

        clients.append(SharedMemClient("trigger", 
                        i, 
                        dtype=dtype))
        clients[i].attach()
        
    for i in range(0, n_samples):
        
        print("###########################")

        for j in range(0, n_envs):

            t = time.perf_counter()
            
            # for i in range(server.tensor_view.shape[0]):

            a = clients[j].read_bool()
            t_end = time.perf_counter() - t 
            print(f"{j} -> time 2 read global bool: " + str(t_end))

            # print(clients[j].tensor_view)

        time.sleep(0.1)

if __name__ == "__main__":
    
    # profile_reading_global_bool()

    # profile_writing_bool_array()

    profile_read_write_cmds_states()
        

        
