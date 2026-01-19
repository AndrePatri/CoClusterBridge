import unittest
import tempfile
from pathlib import Path

from mpc_hive.tests.remote_stepping_fakes import (
    DummyClusterClient,
    write_dummy_srdf,
    NAMESPACE,
)

CLUSTER_SIZE = 1
JOINT_NAMES = ["joint_1", "joint_2"]
CONTACT_NAMES = ["contact_1"]


class RemoteSteppingClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.namespace = NAMESPACE
        self.srdf_path = write_dummy_srdf(Path(self.tmp_dir.name) / "dummy.srdf", JOINT_NAMES)

        self.client = DummyClusterClient(self.namespace, self.srdf_path, JOINT_NAMES, CONTACT_NAMES, CLUSTER_SIZE)

    def tearDown(self):

        self.tmp_dir.cleanup()

    def test_client(self):
        # Blocking run; external orchestration should terminate the client when done.
        self.client.run()

if __name__ == "__main__":
    unittest.main()
