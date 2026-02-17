import unittest
import tempfile
from pathlib import Path

from mpc_hive.tests.remote_stepping_fakes import (
    DummyClusterClient,
    write_dummy_srdf,
    NAMESPACE,
    CLUSTER_SIZE
)

N_JNTS=42
JOINT_NAMES = [f"joint_{i}" for i in range(N_JNTS)] # simulating a client with controllers having a reduced order model wrt server
JOINT_NAMES_PARTIAL = [f"joint_{i}" for i in range(N_JNTS-10)] # simulating a client with controllers having a reduced order model wrt server

CONTACT_NAMES = ["contact_1"]


class RemoteSteppingClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.namespace = NAMESPACE
        self.srdf_path = write_dummy_srdf(Path(self.tmp_dir.name) / "dummy.srdf", JOINT_NAMES)

        self.client = DummyClusterClient(self.namespace, self.srdf_path, JOINT_NAMES_PARTIAL, CONTACT_NAMES, CLUSTER_SIZE)

    def tearDown(self):

        self.tmp_dir.cleanup()

    def test_client(self):
        # Blocking run; external orchestration should terminate the client when done.
        self.client.run()

if __name__ == "__main__":
    unittest.main()
