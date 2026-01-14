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

CONTROL_DT = 0.01
CLUSTER_DT = 0.01
ACK_TIMEOUT_MS = 30000
REGISTER_TIMEOUT_S = 30.0
NAMESPACE = os.environ.get("MPCHIVE_REMOTE_STEP_NS", "mpc_hive_remote_step346")


def write_dummy_srdf(path, joint_names: List[str]) -> str:
    lines = ["<?xml version=\"1.0\"?>", "<robot name=\"dummy\">", "  <group_state name=\"home\" group=\"dummy_group\">"]
    for name in joint_names:
        lines.append(f"    <joint name=\"{name}\" value=\"0.0\"/>")
    lines.extend(["  </group_state>", "</robot>"])
    path.write_text("\n".join(lines))
    return str(path)


def wait_for_cluster_server(namespace: str, cluster_size: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = RhcStatus(
                is_server=False,
                namespace=namespace,
                cluster_size=cluster_size,
                n_contacts=None,
                n_nodes=None,
                optimize_mem=True,
                with_torch_view=False,
                with_gpu_mirror=False,
                force_reconnection=True,
            )
            status.run()
            status.close()
            return True
        except Exception:
            time.sleep(0.1)
    return False


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
            verbose=True,
            vlevel=VLevel.V0,
            debug=False,
            force_reconnection=True,
            timeout_ms=ACK_TIMEOUT_MS,
        )

    def step_once(self):
        self.pre_trigger()
        self.trigger_solution()
        self.wait_for_solution()


class DummyController(RHController):
    def __init__(
        self,
        namespace: str,
        srdf_path: str,
        joint_names: List[str],
        contact_names: List[str],
        cluster_size: int,
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

        super().__init__(
            srdf_path=srdf_path,
            n_nodes=3,
            dt=CONTROL_DT,
            namespace=namespace,
            dtype=np.float32,
            verbose=True,
            debug=False,
            timeout_ms=ACK_TIMEOUT_MS,
        )

    def _close(self):
        if not self._closed:
            super()._close()

    def _reset(self):
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

    def _get_jnt_q_from_sol(self, node_idx=1) -> np.ndarray:
        return self._zero_jnts

    def _get_jnt_v_from_sol(self, node_idx=1) -> np.ndarray:
        return self._zero_jnts

    def _get_jnt_a_from_sol(self, node_idx=0) -> np.ndarray:
        return self._zero_jnts

    def _get_jnt_eff_from_sol(self, node_idx=0) -> np.ndarray:
        return self._zero_jnts

    def _get_root_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        return self._zero_root_q

    def _get_full_q_from_sol(self, node_idx=1) -> np.ndarray:
        return self._full_q

    def _get_root_twist_from_sol(self, node_idx=1) -> np.ndarray:
        return self._zero_root_twist

    def _get_root_a_from_sol(self, node_idx=0) -> np.ndarray:
        return self._zero_root_a

    def _update_open_loop(self):
        return None

    def _update_closed_loop(self):
        return None

    def _solve(self) -> bool:
        self._steps += 1
        return True

    def _get_ndofs(self):
        return self.n_dofs

    def _get_robot_mass(self):
        return 1.0

    def _init_problem(self):
        self.n_dofs = len(self._joint_names)
        self.n_contacts = len(self._contact_names)
        self._assign_controller_side_jnt_names(self._joint_names)
        self._zero_jnts = np.zeros((1, self.n_dofs), dtype=self._dtype)
        self._zero_root_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=self._dtype)
        self._zero_root_twist = np.zeros((1, 6), dtype=self._dtype)
        self._zero_root_a = np.zeros((1, 6), dtype=self._dtype)
        self._full_q = np.concatenate([self._zero_root_q, self._zero_jnts], axis=1)

    def _post_problem_init(self):
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
            debug=False,
            custom_opts={},
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
        if int(status.controllers_counter.get_numpy_mirror()[0, 0]) >= expected:
            return True
        time.sleep(0.1)
    return False
