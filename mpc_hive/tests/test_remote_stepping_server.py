import unittest

from mpc_hive.tests.remote_stepping_fakes import (
    DummyClusterServer,
    REGISTER_TIMEOUT_S,
    wait_for_controllers,
    NAMESPACE,
)

CLUSTER_SIZE = 10
N_STEPS = 1000
JOINT_NAMES = ["joint_1", "joint_2"]
CONTACT_NAMES = ["contact_1"]
JOIN_TIMEOUT_S = 5.0


class RemoteSteppingServerTests(unittest.TestCase):
    def setUp(self):
        self.namespace = NAMESPACE
        # server side does not need an srdf file; only clients do

        self.server = DummyClusterServer(self.namespace, CLUSTER_SIZE, JOINT_NAMES, CONTACT_NAMES)
        self.server.run()

    def tearDown(self):
        try:
            self.server.close()
        except Exception:
            pass

    def test_remote_stepping_server_steps_cluster(self):
        self.assertTrue(wait_for_controllers(self.server, CLUSTER_SIZE, REGISTER_TIMEOUT_S))
        for _ in range(N_STEPS):
            self.server.step_once()
        self.assertEqual(self.server.solution_counter(), N_STEPS)


if __name__ == "__main__":
    unittest.main()
