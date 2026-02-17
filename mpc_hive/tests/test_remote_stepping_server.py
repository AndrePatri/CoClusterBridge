import unittest

from mpc_hive.tests.remote_stepping_fakes import (
    DummyClusterServer,
    NAMESPACE,
    N_PHYSICS_STEPS,
    CLUSTER_SIZE
)
import numpy as np

N_STEPS = 1000
N_JNTS=42
JOINT_NAMES = ["joint_{}".format(i) for i in range(N_JNTS)]
CONTACT_NAMES = ["contact_1"]

class RemoteSteppingServerTests(unittest.TestCase):
    def setUp(self):
        self.namespace = NAMESPACE
        # server side does not need an srdf file; only clients do

        self.server = DummyClusterServer(self.namespace, CLUSTER_SIZE, JOINT_NAMES, CONTACT_NAMES)
        self.server.run()

    def tearDown(self):
        self.server.close()

    def _read_state_from_robot(self):
        # Mimics world_interface_base warmup: write dummy simulator state and refs.
        state = self.server.get_state()

        # read from sensors
        dummy_joint_state=np.zeros((CLUSTER_SIZE, state.jnts_state.n_jnts))
        dummy_q=np.zeros((CLUSTER_SIZE, 7))
        dummy_base_twist=np.zeros((CLUSTER_SIZE, 6))
        dummy_gravity=np.zeros((CLUSTER_SIZE, 3))
        dummy_gravity[:,2]=-9.81

        # set and write to shared memory
        state.jnts_state.set(data=dummy_joint_state, data_type="q", robot_idxs=None)
        state.jnts_state.set(data=dummy_joint_state, data_type="v", robot_idxs=None)
        state.jnts_state.set(data=dummy_joint_state, data_type="a", robot_idxs=None)
        state.jnts_state.set(data=dummy_joint_state, data_type="eff", robot_idxs=None)
        state.root_state.set(data=dummy_q, data_type="q_full", robot_idxs=None)
        state.root_state.set(data=dummy_base_twist, data_type="twist", robot_idxs=None)
        state.root_state.set(data=dummy_gravity, data_type="gn", robot_idxs=None)

        self.server.write_robot_state()

    def _set_refs_for_mpcs(self):
        # write dummy refs for MPCs to track (this depends on specific MPC implementation)
        refs = self.server.get_refs()
        p_ref_dummy=np.zeros((CLUSTER_SIZE, 3))
        q_ref_dummy=np.zeros((CLUSTER_SIZE, 4))
        q_ref_dummy[:,3]=1.0
        twist_ref_dummy=np.zeros((CLUSTER_SIZE, 6))
        refs.rob_refs.root_state.set(data_type="p", data=p_ref_dummy, robot_idxs=None)
        refs.rob_refs.root_state.set(data_type="q", data=q_ref_dummy, robot_idxs=None)
        refs.rob_refs.root_state.set(data_type="twist", data=twist_ref_dummy, robot_idxs=None)

        refs.rob_refs.root_state.synch_all(read=False, retry=True) # write everything

    def _activate_all_controllers(self):
        status = self.server.get_status()
        status.activation_state.synch_all(read=True, retry=True)
        status.activation_state.get_numpy_mirror()[:, :] = True
        status.activation_state.synch_all(read=False, retry=True)

    def _set_cluster_actions(self):
        actions = self.server.get_actions()
        actions.jnts_state.synch_all(read=True, retry=True)
        pos_ref=actions.jnts_state.get(data_type="q", gpu=False)
        vel_ref=actions.jnts_state.get(data_type="v", gpu=False)
        eff_ref=actions.jnts_state.get(data_type="eff", gpu=False)
    
        # here MPC sol. should be used to set references for the lower-level controller (e.g. joint impedance).

    def _step_world(self):
        # simulation physics step or nothing if in real world
        pass
    
    def _reset(self, failed):
        # any reset logic

        self.server.reset_controllers(idxs=failed)

        self._read_state_from_robot()

        self.server.activate_controllers(idxs=self.server.get_inactive_controllers())

    def test_remote_stepping_server_steps_cluster(self):
        
        self._read_state_from_robot()
        
        self._activate_all_controllers()

        # initial trigger
        self.server.pre_trigger() # read controllers state
        self._set_refs_for_mpcs() # write dummy refs for MPCs to track
        self.server.trigger_solution() # send solution trigger signal

        # example loop for server-client stepping
        physics_steps = 0
        for i in range(N_PHYSICS_STEPS*N_STEPS):
            if self.server.is_cluster_instant(physics_steps):
                wait_ok=self.server.wait_for_solution() # blocking, wait for last solution
                if not wait_ok:
                    break
                failed = self.server.get_failed_controllers()
                self._set_cluster_actions() # write MPC cmds to low-level controllers
                
                if self.server.solution_counter() >= N_STEPS:
                    break

                self._read_state_from_robot() # read updated state from robot and write to cluster
                if failed is not None:
                    self._reset()

                self._set_refs_for_mpcs() # write dummy refs for MPCs to track

                self.server.pre_trigger()
                self.server.trigger_solution() # non-blocking

            self._step_world() # physics advances in parallel with MPC solution (mimics real-world behavior)
            physics_steps+=1
        
        print(f"Performed {self.server.solution_counter()} remote cluster steps. Target was {N_STEPS}.")
        self.assertEqual(self.server.solution_counter(), N_STEPS)


if __name__ == "__main__":
    unittest.main()
