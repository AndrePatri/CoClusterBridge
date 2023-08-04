import torch

from abc import ABC, abstractmethod

from control_cluster_utils.utilities.control_cluster_utils import RobotClusterState, RobotClusterCmd, ActionChild
from control_cluster_utils.utilities.pipe_utils import NamedPipesHandler
OMode = NamedPipesHandler.OMode
DSize = NamedPipesHandler.DSize

import os
import struct

import time

import numpy as np

import multiprocess as mp

from typing import List

class ControlClusterClient(ABC):

    def __init__(self, 
            cluster_size: int, 
            control_dt: float,
            cluster_dt: float,
            jnt_names: List[str],
            backend = "torch", 
            device = torch.device("cpu"), 
            np_array_dtype = np.float32, 
            verbose = False):

        self._verbose = verbose

        self.np_dtype = np_array_dtype
        data_aux = np.zeros((1, 1), dtype=self.np_dtype)
        self.np_array_itemsize = data_aux.itemsize

        if self.np_dtype == np.float64:
            self.torch_dtype = torch.float64
        if self.np_dtype == np.float32:
            self.torch_dtype = torch.float32

        self.jnt_names = jnt_names

        self.n_dofs = len(self.jnt_names)
        self.jnt_data_size = np.zeros((self.n_dofs, 1),
                                    dtype=self.np_dtype).nbytes
        
        self.cluster_size = cluster_size
        
        self.jnt_names = jnt_names

        self.cluster_dt = cluster_dt # dt at which the controllers in the cluster will run 
        self.control_dt = control_dt # dt at which the low level controller or the simulator runs

        self._backend = backend
        self._device = device

        self.robot_states = RobotClusterState(self.n_dofs, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype) # from robot to controllers
        
        self.controllers_cmds = RobotClusterCmd(self.n_dofs, 
                                            cluster_size=self.cluster_size, 
                                            backend=self._backend, 
                                            device=self._device, 
                                            dtype=self.torch_dtype)

        self.is_cluster_ready = mp.Value('b', False)
        self._trigger_solve = mp.Array('b', self.cluster_size)
        self._trigger_read = mp.Array('b', self.cluster_size)
        self._trigger_send_state = mp.Array('b', self.cluster_size)

        self.status = "status"
        self.info = "info"
        self.warning = "warning"
        self.exception = "exception"
    
        self.pipes_manager = NamedPipesHandler()
        self.pipes_manager.create_buildpipes()
        self.pipes_manager.create_runtime_pipes(self.cluster_size) # we create the remaining pipes

        self.solution_time = -1.0
        self.n_sim_step_per_cntrl = -1
        self.solution_counter = 0
        self._compute_n_control_actions()

        self._create_shared_buffers() # used to store the data received by the pipes 
        # on the child processes

        self._spawn_processes() # we launch all the child processes

    def _spawn_processes(self):
        
        # we spawn the handshake() to another process, 
        # so that it's not blocking
        self._connection_process = mp.Process(target=self._handshake, 
                                name = "ControlClusterClient_handshake")
        self._connection_process.start()
        print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _handshake process")

        self._trigger_processes = []
        self._solread_processes = []
        self._statesend_processes = []

        for i in range(0, self.cluster_size):

            self._trigger_processes.append(mp.Process(target=self._trigger_solution, 
                                                    name = "ControlClusterClient_trigger" + str(i), 
                                                    args=(i, )), 
                                )
            print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": spawned _trigger_solution process n." + str(i))
            self._trigger_processes[i].start()

            self._solread_processes.append(mp.Process(target=self._read_solution, 
                                                    name = "ControlClusterClient_solread" + str(i), 
                                                    args=(i, )), 
                                )
            print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _read_solution process n." + str(i))
            self._solread_processes[i].start()

            self._statesend_processes.append(mp.Process(target=self._send_states, 
                                                    name = "ControlClusterClient_sendstates" + str(i), 
                                                    args=(i, )), 
                                )
            print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": spawned _send_states process n." + str(i))
            self._statesend_processes[i].start()

    def _handshake(self):
        
        # THIS RUNS IN A CHILD PROCESS --> we perform the "handshake" with
        # the server: we exchange crucial info which has to be shared between 
        # them
        
        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": waiting for handshake with the ControlCluster server...")

        # retrieves/sends some important configuration information from the server

        # cluster size TO cluster server
        self.pipes_manager.open_pipes(["cluster_size"], 
                                    mode=OMode["O_WRONLY"])
        cluster_size_data = struct.pack('i', self.cluster_size)
        os.write(self.pipes_manager.pipes_fd["cluster_size"], cluster_size_data) # the server is listening -> we send the info we need

        # robot joint number FROM server
        self.pipes_manager.open_pipes(selector=["jnt_number_srvr"], 
                                mode=OMode["O_RDONLY"])
        jnt_number_raw = os.read(self.pipes_manager.pipes_fd["jnt_number_srvr"], DSize["int"])
        n_dofs_srvr_side = struct.unpack('i', jnt_number_raw)[0]

        if len(self.jnt_names) != n_dofs_srvr_side:

            exception = f"[{self.__class__.__name__}]" + f"{self.exception}" + \
                f": the received number of robot joints from cluster server {n_dofs_srvr_side} " + \
                f"does not match the client-side value of {len(self.jnt_names)}"
            
            raise Exception(exception)

        # client-side robot joints name TO cluster server
        # we first encode the list
        jnt_names_client_string = "\n".join(self.jnt_names)  # convert list to a newline-separated string
        jnt_names_client_raw = jnt_names_client_string.encode('utf-8')
        n_bytes_jnt_names = len(jnt_names_client_raw)

        self.pipes_manager.open_pipes(["n_bytes_jnt_names_client"], 
                                    mode=OMode["O_WRONLY"])
        n_bytes_jnt_names_raw = struct.pack('i', n_bytes_jnt_names)
        os.write(self.pipes_manager.pipes_fd["n_bytes_jnt_names_client"], n_bytes_jnt_names_raw)

        self.pipes_manager.open_pipes(["jnt_names_client"], 
                                    mode=OMode["O_WRONLY"])
        
        os.write(self.pipes_manager.pipes_fd["jnt_names_client"], 
                jnt_names_client_raw)

        self.is_cluster_ready.value = True # we signal the main process
        # the connection is established

        print(f"[{self.__class__.__name__}]" + f"{self.info}" + ": friendship with ControlCluster server established.")

    def _trigger_solution(self, 
                        index: int):

        # solver
        self.pipes_manager.open_pipes(["trigger_solve"], 
                                    mode=OMode["O_WRONLY"], 
                                    index=index) # blocking (non-blocking
        # would throw error if nothing has opened the pipe in read mode)
        
        while True: # we keep the process alive

            if self._trigger_solve[index] and self.is_cluster_ready.value: # these are set by the parent process
                
                msg_bytes = b's'
                # Send a signal to perform the solution
                os.write(self.pipes_manager.pipes_fd["trigger_solve"][index], 
                    msg_bytes)

                self._trigger_solve[index] = False # we will wait for next signal
                # from the main process
            
            else:

                continue

    def _read_solution(self, 
                    index: int):

        # these are not blocking
        self.pipes_manager.open_pipes(selector=[
                "cmd_jnt_q", "cmd_jnt_v", "cmd_jnt_eff", 
                "rhc_info"
                ], 
                mode = OMode["O_RDONLY"], 
                index=index)

        while True: # we keep the process alive
            
            if self._trigger_read[index] and self.is_cluster_ready.value: # these are set by the parent process

                self._cmd_q_buffer[(index * self.n_dofs):(index * self.n_dofs + self.n_dofs)] = \
                                np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_q"][index], self.jnt_data_size), 
                                                dtype=self.np_dtype).reshape((1, self.n_dofs)).flatten()
                        
                self._cmd_v_buffer[(index * self.n_dofs):(index * self.n_dofs + self.n_dofs)] = \
                    np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_v"][index], self.jnt_data_size),
                                    dtype=self.np_dtype).reshape((1, self.n_dofs)).flatten()
                
                self._cmd_eff_buffer[(index * self.n_dofs):(index * self.n_dofs + self.n_dofs)] = \
                    np.frombuffer(os.read(self.pipes_manager.pipes_fd["cmd_jnt_eff"][index], self.jnt_data_size),
                                    dtype=self.np_dtype).reshape((1, self.n_dofs)).flatten()
                
                self._rhc_info_buffer[(index * self._add_info_size):(index * self._add_info_size + self._add_info_size)] = \
                    np.frombuffer(os.read(self.pipes_manager.pipes_fd["rhc_info"][index], self._add_info_datasize),
                                    dtype=self.np_dtype)

                self._trigger_read[index] = False # we will wait for next signal
                # from the main process
            
            else:

                continue
    
    def _send_states(self, 
                    index: int):

        self.pipes_manager.open_pipes(selector=["state_root_p", "state_root_q", 
                                                "state_root_v", "state_root_omega", 
                                                "state_jnt_q", "state_jnt_v"
                                                ], 
                                                mode = OMode["O_WRONLY"], 
                                                index=index)

        root_p_size = self.robot_states.root_state.p.shape[1]

        root_q_size = self.robot_states.root_state.q.shape[1]
        root_v_size = self.robot_states.root_state.v.shape[1]
        root_omega_size = self.robot_states.root_state.omega.shape[1]
        jnt_q_size = self.robot_states.jnt_state.q.shape[1]
        jnt_v_size = self.robot_states.jnt_state.v.shape[1]

        while True: # we keep the process alive

            if self._trigger_send_state[index] and self.is_cluster_ready.value: # these are set by the parent process
                
                # root state
                os.write(self.pipes_manager.pipes_fd["state_root_p"][index], 
                        np.array(self._state_root_p_buffer, 
                                dtype=self.np_dtype)[(index * root_p_size):(index * root_p_size + root_p_size)].tobytes())

                os.write(self.pipes_manager.pipes_fd["state_root_q"][index], 
                        np.array(self._state_root_q_buffer, 
                                dtype=self.np_dtype)[(index * root_q_size):(index * root_q_size + root_q_size)].tobytes())
                os.write(self.pipes_manager.pipes_fd["state_root_v"][index], 
                        np.array(self._state_root_v_buffer, 
                                dtype=self.np_dtype)[(index * root_v_size):(index * root_v_size + root_v_size)].tobytes())
                os.write(self.pipes_manager.pipes_fd["state_root_omega"][index],
                        np.array(self._state_root_omega_buffer, 
                                dtype=self.np_dtype)[(index * root_omega_size):(index * root_omega_size + root_omega_size)].tobytes())

                # jnt state
                os.write(self.pipes_manager.pipes_fd["state_jnt_q"][index], 
                        np.array(self._state_jnt_q_buffer, 
                                dtype=self.np_dtype)[(index * jnt_q_size):(index * jnt_q_size + jnt_q_size)].tobytes())
                os.write(self.pipes_manager.pipes_fd["state_jnt_v"][index], 
                        np.array(self._state_jnt_v_buffer,
                                dtype=self.np_dtype)[(index * jnt_v_size):(index * jnt_v_size + jnt_v_size)].tobytes())

                self._trigger_send_state[index] = False # we will wait for next signal
                # from the main process
            
            else:

                continue
    
    def _create_shared_buffers(self):
        
        buffer_cdtype = np.ctypeslib.as_ctypes_type(self.np_dtype)

        # cmds from controllers to robot
        self._cmd_q_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.n_dofs)
        self._cmd_v_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.n_dofs)
        self._cmd_eff_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.n_dofs) 

        self._add_info_size = 2
        self._add_info_datasize = 2 * self.np_array_itemsize

        # additional info from controllers
        self._rhc_info_buffer = mp.Array(buffer_cdtype, 
                                        self.robot_states.cluster_size * self._add_info_size) 

        # state from robot to controllers
        self._state_root_p_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.robot_states.root_state.p.shape[1])
        self._state_root_q_buffer = mp.Array(buffer_cdtype,
                                    self.robot_states.cluster_size * self.robot_states.root_state.q.shape[1])
        self._state_root_v_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.robot_states.root_state.v.shape[1])
        self._state_root_omega_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.robot_states.root_state.omega.shape[1])
        self._state_jnt_q_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.robot_states.jnt_state.q.shape[1])
        self._state_jnt_v_buffer = mp.Array(buffer_cdtype, 
                                    self.robot_states.cluster_size * self.robot_states.jnt_state.v.shape[1])

    def _fill_cmds_from_buffer(self):

        # HERE IS WHERE COPIES FROM CPU TO GPU ARE MADE (if the used device is GPU)

        self.controllers_cmds.jnt_cmd.q = torch.frombuffer(self._cmd_q_buffer.get_obj(), 
                    dtype=self.controllers_cmds.dtype).reshape(self.controllers_cmds.cluster_size, 
                                                                self.controllers_cmds.n_dofs).to(self._device)
        
        self.controllers_cmds.jnt_cmd.v = torch.frombuffer(self._cmd_v_buffer.get_obj(), 
                    dtype=self.controllers_cmds.dtype).reshape(self.controllers_cmds.cluster_size, 
                                                                self.controllers_cmds.n_dofs).to(self._device)

        self.controllers_cmds.jnt_cmd.eff = torch.frombuffer(self._cmd_eff_buffer.get_obj(), 
                    dtype=self.controllers_cmds.dtype).reshape(self.controllers_cmds.cluster_size, 
                                                                self.controllers_cmds.n_dofs).to(self._device)
        
        self.controllers_cmds.rhc_info.data = torch.frombuffer(self._rhc_info_buffer.get_obj(),
                    dtype=self.controllers_cmds.dtype).reshape(self.controllers_cmds.cluster_size, 
                                                                self._add_info_size).to(self._device)
    
    def _fill_buffers_with_states(self):
        
        # HERE IS WHERE COPIES OF THE STATE FROM GPU TO CPU ARE MADE (if the used device is CPU)

        # root state 
        self._state_root_p_buffer[:] = self.robot_states.root_state.p.cpu().flatten(start_dim=0).numpy() # we flatten along clusters
        
        self._state_root_q_buffer[:] = self.robot_states.root_state.q.cpu().flatten(start_dim=0).numpy()
        
        self._state_root_v_buffer[:] = self.robot_states.root_state.v.cpu().flatten(start_dim=0).numpy()
        
        self._state_root_omega_buffer[:] = self.robot_states.root_state.omega.cpu().flatten(start_dim=0).numpy()
        
        # jnt state
        self._state_jnt_q_buffer[:] = self.robot_states.jnt_state.q.cpu().flatten(start_dim=0).numpy()
        
        self._state_jnt_v_buffer[:] = self.robot_states.jnt_state.v.cpu().flatten(start_dim=0).numpy()
    
    def _send_sol_trigger(self):

        for i in range(0, self.cluster_size):

            self._trigger_solve[i] = True # we signal the solution triggger process n.{i} to trigger the solution 
            # of the associated controller

    def _send_state_trigger(self):

        for i in range(0, self.cluster_size):

            self._trigger_send_state[i] = True # we signal the state process n.{i} to send the state to the 
            # associated controller

    def _send_read_sols_trigger(self):

        for i in range(self.cluster_size):
            
            self._trigger_read[i] = True

    def _wait_for_solutions(self):
        
        while not all(not value for value in self._trigger_read):

            continue
    
    def _wait_for_state_writing(self):

        while not all(not value for value in self._trigger_send_state):

            continue

    def _compute_n_control_actions(self):

        if self.cluster_dt < self.control_dt:

            print(f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + ": cluster_dt has to be >= control_dt")

            self.n_sim_step_per_cntrl = 1
        
        else:
            
            self.n_sim_step_per_cntrl = round(self.cluster_dt / self.control_dt)
            self.cluster_dt = self.control_dt * self.n_sim_step_per_cntrl

        message = f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": the cluster controllers will run at a rate of " + \
                str(1.0 / self.cluster_dt) + " Hz"\
                ", while the low level control will run at " + str(1.0 / self.control_dt) + "Hz.\n" + \
                "Number of sim steps per control steps: " + str(self.n_sim_step_per_cntrl)

        print(message)
    
    def is_cluster_instant(self, 
                        control_index: int):
        
        # control_index is, e.g., the current simulation loop number (0-based)

        return (control_index+1) % self.n_sim_step_per_cntrl == 0
    
    def solve(self):

        # solve all the TO problems in the control cluster

        if not self.is_cluster_ready.value:

            if self._verbose: 

                print(f"[{self.__class__.__name__}]"  + f"[{self.status}]" + ": waiting connection to ControlCluster server")

        if (self.is_cluster_ready.value):
            
            start_time = time.time() # we profile the whole solution pipeline
            
            self._fill_buffers_with_states() # we fill the buffers with the states

            self._send_state_trigger() # we send the state to each controller
            
            self._wait_for_state_writing() # we wait until everything was sent (as soon as each controller 
            # receives a state, it solves the TO)

            self._send_sol_trigger() # we send a signal to solve the TO to all controllers

            self._send_read_sols_trigger() # reads from all controllers' solutions (this will automatically update the shared
            # # data buffers)
            
            self._wait_for_solutions() # will wait until all controllers are done solving their TO
            
            self._fill_cmds_from_buffer() # copies data from the solution buffers to the robot cmds object
            
            self.solution_counter += 1

            self.solution_time = time.time() - start_time # we profile the whole solution pipeline

    def close(self):

        self.__del__()
    
    def _close_process(self, 
                    process):
        
        if process.is_alive():
            
            process.terminate()  # Forcefully terminate the process
                    
            print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": terminating child process " + str(process.name))
        
            process.join()
        
    def __del__(self):
        
        if self._connection_process is not None:
            
            self._close_process(self._connection_process)

        for process in self._trigger_processes:

            self._close_process(process)

        for process in self._solread_processes:

            self._close_process(process)
        
        for process in self._statesend_processes:

            self._close_process(process)