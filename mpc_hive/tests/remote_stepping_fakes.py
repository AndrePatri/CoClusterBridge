import os
import time
from typing import List

import numpy as np

from EigenIPC.PyEigenIPC import VLevel

from mpc_hive.cluster_client.control_cluster_client import ControlClusterClient
from mpc_hive.cluster_server.control_cluster_server import ControlClusterServer
from mpc_hive.controllers.rhc import RHController
from mpc_hive.utilities.shared_data.rhc_data import RhcStatus
from mpc_hive.utilities.shared_data.rhc_data import RhcRefs

CONTROL_DT = 0.0001 # dt at which lower lever controllers runs (e.g. joint impedance controllers)
CLUSTER_DT = 0.03 # dt at which the cluster server steps (MPC dt)
N_PHYSICS_STEPS=int(CLUSTER_DT / CONTROL_DT)
ACK_TIMEOUT_MS = 8000
N_NODES=30
NAMESPACE = "mpc_hive_test_ns_"

def write_dummy_srdf(path, joint_names: List[str]) -> str:
    lines = ["<?xml version=\"1.0\"?>", "<robot name=\"dummy\">", "  <group_state name=\"home\" group=\"dummy_group\">"]
    for name in joint_names:
        lines.append(f"    <joint name=\"{name}\" value=\"0.0\"/>")
    lines.extend(["  </group_state>", "</robot>"])
    path.write_text("\n".join(lines))
    return str(path)

class DummyClusterServer(ControlClusterServer):
    def __init__(self, namespace: str, cluster_size: int, joint_names: List[str], contact_names: List[str]):
        super().__init__(
            namespace=namespace,
            cluster_size=cluster_size,
            control_dt=CONTROL_DT,
            cluster_dt=CLUSTER_DT,
            jnt_names=joint_names,
            n_contacts=len(contact_names),
            contact_linknames=contact_names,
            use_gpu=False,
            use_torch=False,
            verbose=True,
            vlevel=VLevel.V0,
            debug=False,
            force_reconnection=True,
            timeout_ms=ACK_TIMEOUT_MS,
        )

    def step_once(self):
        self.pre_trigger()  # retrieves current controllers status 
        # (here custom logic depending on controllers status could be added)
        self.trigger_solution() # sends trigger to controllers
        self.wait_for_solution() # waits for ALL controllers to solve

class DummyController(RHController):
    def __init__(
        self,
        namespace: str,
        srdf_path: str,
        joint_names: List[str],
        contact_names: List[str],
        cluster_size: int,
        closed_loop: bool = True,
    ):
        self._joint_names = list(joint_names)
        self._contact_names = list(contact_names)
        self._cluster_size = cluster_size
        self._zero_jnts = None
        self._zero_root_q = None
        self._zero_root_twist = None
        self._zero_root_a = None
        self._full_q = None
        self._steps = 0

        self._closed_loop=closed_loop

        super().__init__(
            srdf_path=srdf_path,
            n_nodes=N_NODES,
            dt=CLUSTER_DT,
            namespace=namespace,
            dtype=np.float32,
            verbose=True,
            debug=True,
            timeout_ms=ACK_TIMEOUT_MS,
        )

    def _close(self):
        if not self._closed:
            super()._close()

    def _reset(self):
        # custom reset logic for controller
        return None

    def _init_rhc_task_cmds(self):
        refs = RhcRefs(
            namespace=self.namespace,
            is_server=False,
            n_robots=1,
            n_jnts=self.n_dofs,
            n_contacts=self.n_contacts,
            jnt_names=self._joint_names,
            contact_names=self._contact_names,
            with_gpu_mirror=False,
            with_torch_view=False,
            force_reconnection=False,
            safe=False,
            verbose=False,
            vlevel=VLevel.V0,
            optimize_mem=True,
        )
        refs.run()
        return refs

    def _get_robot_jnt_names(self):
        return self._joint_names

    def _get_contact_names(self):
        return self._contact_names

    def _get_ndofs(self):
        return self.n_dofs

    def _get_robot_mass(self):
        return 60.0
    
    def _get_jnt_q_from_sol(self, node_idx=1) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_jnts[:, node_idx:node_idx + 1].T

    def _get_jnt_v_from_sol(self, node_idx=1) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_jnts[:, node_idx:node_idx + 1].T

    def _get_jnt_a_from_sol(self, node_idx=0) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_jnts[:, node_idx:node_idx + 1].T

    def _get_jnt_eff_from_sol(self, node_idx=0) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_jnts[:, node_idx:node_idx + 1].T

    def _get_root_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_root_q[:, node_idx:node_idx + 1].T

    def _get_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._full_q[:, node_idx:node_idx + 1].T

    def _get_root_twist_from_sol(self, node_idx=1) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_root_twist[:, node_idx:node_idx + 1].T

    def _get_root_a_from_sol(self, node_idx=0) -> np.ndarray:
        # In a real controller these values are read from the solver solution.
        return self._zero_root_a[:, node_idx:node_idx + 1].T

    def _get_cost_info(self):
        
        # dummy cost
        cost_dict = self.cost_dict_dummy.copy()
        cost_names = list(cost_dict.keys())
        cost_dims = [1] * len(cost_names) # costs are always scalar
        return cost_names, cost_dims
    
    def _get_constr_info(self):
        
        constr_dict = self.constr_dict_dummy.copy()
        
        constr_names = list(constr_dict.keys())
        constr_dims = [-1] * len(constr_names)
        i = 0
        for constr in constr_dict:
            constr_val = constr_dict[constr]
            constr_shape = constr_val.shape
            constr_dims[i] = constr_shape[0]
            i+=1
        return constr_names, constr_dims
    
    def _get_cost_from_sol(self,
                    cost_name: str):
        return self.rhc_costs[cost_name]
    
    def _get_constr_from_sol(self,
                    constr_name: str):
        return self.rhc_constr[constr_name]
    
    def _update_open_loop(self):
        # set initial guess and initial states for controller
        # by just using the last solution, no feedback from real world
        return None

    def _update_closed_loop(self):
        # set initial guess and initial states for controller
        # by reading measured robot data
        return None

    def _update_db_data(self):
        
        # add profiling data to profiling dict
        self._profiling_data_dict["some_metric"] = 1.2345

        self.rhc_costs.update(self.cost_dict_dummy)
        self.rhc_constr.update(self.constr_dict_dummy)

    def _rti(self):
        # solve controller problem with in real-time iteration 
        time.sleep(0.003)  # simulate some solving time

    def _solve(self) -> bool:
        if self._closed_loop:
            self._update_closed_loop()
            print(f"Controller n. {self.controller_index}: problem solved (closed loop).")
        else:
            self._update_open_loop()
            print(f"Controller n. {self.controller_index}: problem solved (open loop).")
        
        self._rti()

        self._update_db_data()

        self._steps += 1
        return True

    def _init_problem(self):
        # initialize problem-> solver-depedent stuff here
        self.n_dofs = len(self._joint_names)
        self.n_contacts = len(self._contact_names)
        self._assign_controller_side_jnt_names(self._joint_names)
        # In a real controller these arrays come from the MPC solver state (one column per node).
        self._zero_jnts = np.zeros((self.n_dofs, self._n_nodes), dtype=self._dtype)
        self._zero_root_q = np.tile(
            np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], dtype=self._dtype),
            (1, self._n_nodes),
        ) # assuming x, y, z, w quaternion order for solver here
        self._zero_root_twist = np.zeros((6, self._n_nodes), dtype=self._dtype)
        self._zero_root_a = np.zeros((6, self._n_nodes-1), dtype=self._dtype)
        self._full_q = np.concatenate([self._zero_root_q, self._zero_jnts], axis=0)

        # e.g. floating base + joints
        self.nq=7+self.n_dofs
        self.nv=6+self.n_dofs

        self.constr_dict_dummy = {"floating_base_inverse_dyn": np.zeros((6, N_NODES)), 
                "integrator": np.zeros((self.nq+self.nv, N_NODES)),
                "init_state": np.zeros((self.nq+self.nv, 1)),
                "other_constr": np.zeros((8, 7))}
        
        self.cost_dict_dummy = {"postural_cost": np.zeros((1, N_NODES-1)), 
                "other_cost": np.zeros((1, N_NODES-1))}
        
    def _post_problem_init(self):

        self.rhc_costs={}
        self.rhc_constr={}        

        return None

class DummyClusterClient(ControlClusterClient):
    def __init__(
        self,
        namespace: str,
        srdf_path: str,
        joint_names: List[str],
        contact_names: List[str],
        cluster_size: int,
    ):
        self._srdf_path = srdf_path
        self._joint_names = list(joint_names)
        self._contact_names = list(contact_names)
        super().__init__(
            namespace=namespace,
            cluster_size=cluster_size,
            processes_basename="DummyController",
            set_affinity=False,
            use_mp_fork=True,
            isolated_cores_only=False,
            verbose=True,
            debug=True,
            custom_opts={"n_nodes": N_NODES, "cluster_dt": CLUSTER_DT, "some_other_mpc_opts": 12345},
        )

    def _generate_controller(self, idx: int):
        return DummyController(
            namespace=self._namespace,
            srdf_path=self._srdf_path,
            joint_names=self._joint_names,
            contact_names=self._contact_names,
            cluster_size=self.cluster_size,
        )

def wait_for_controllers(server: ControlClusterServer, expected: int, timeout_s: float) -> bool:
    status = server.get_status()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status.controllers_counter.synch_all(read=True, retry=True)
        if int(status.controllers_counter.get_numpy_mirror()[0, 0]) == expected:
            return True
        time.sleep(0.1)
    return False
