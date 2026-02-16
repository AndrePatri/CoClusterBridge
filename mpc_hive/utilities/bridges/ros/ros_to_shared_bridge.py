from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.from_ros import FromRos
from EigenIPC.PyEigenIPC import ClientFactory
from EigenIPC.PyEigenIPC import VLevel, LogType, Journal, dtype

import argparse
import time
import numpy as np

from perf_sleep.pyperfsleep import PerfSleep
from mpc_hive.utilities.sysutils import set_process_affinity


class RosToSharedMemBridge:

    _CATALOG_BASENAME = "RosBridgeCatalog"

    def __init__(self,
            namespace: str,
            backend: str = "ros2",
            add_training_data: bool = False,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            force_reconnection: bool = True,
            remap_ns: str = None):

        self._namespace = namespace
        self._backend = backend
        self._add_training_data = add_training_data  # reserved for compatibility
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size
        self._force_reconnection = force_reconnection
        self._remap_ns = remap_ns

        self._bridges = {}

        self._catalog_bridge = None
        self._catalog_client = None
        self._catalog_buffer = None

        self._dt = 0.05
        self._is_running = False
        self._node = None

        self._warn_counter = 0

        self._check_backend()

    def _check_backend(self):

        if self._backend not in ("ros1", "ros2"):
            Journal.log(self.__class__.__name__,
                "_check_backend",
                f"backend {self._backend} not supported!",
                LogType.EXCEP,
                throw_when_excep=True)

    def _backend_alive(self):

        if self._backend == "ros1":
            import rospy
            return not rospy.is_shutdown()

        if self._backend == "ros2":
            import rclpy
            return rclpy.ok()

        return True

    def _shutdown_backend(self):

        if self._backend == "ros1":
            try:
                import rospy
                if not rospy.is_shutdown():
                    rospy.signal_shutdown("bridge close requested")
            except Exception:
                pass

        if self._backend == "ros2":
            try:
                import rclpy
                if self._node is not None:
                    self._node.destroy_node()
                    self._node = None
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def _spin_backend_once(self):

        if self._backend == "ros2" and self._backend_alive():
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=0.0)

    def _init_backend(self):

        if self._backend == "ros1":
            import rospy
            rospy.init_node("Ros2SharedMemoryBridge_" + self._namespace)
        elif self._backend == "ros2":
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("Ros2SharedMemoryBridge_" + self._namespace)

    def _remap_namespace(self, namespace):

        if self._remap_ns is None:
            return namespace

        return namespace.replace(self._namespace, self._remap_ns)

    def _init_catalog_bridge(self):

        kwargs = dict(
            basename=self._CATALOG_BASENAME,
            namespace=self._namespace,
            queue_size=self._queue_size,
            ros_backend=self._backend,
            vlevel=self._vlevel,
            verbose=self._verbose,
            force_reconnection=self._force_reconnection,
            remap_ns=self._remap_namespace(self._namespace),
        )
        if self._backend == "ros2":
            kwargs["node"] = self._node

        self._catalog_bridge = FromRos(**kwargs)

    def _wait_catalog_bridge_ready(self):

        warn_counter = 0

        while self._backend_alive():
            self._spin_backend_once()

            if self._catalog_bridge.run():
                return True

            if warn_counter % 20 == 0:
                Journal.log(self.__class__.__name__,
                    "_wait_catalog_bridge_ready",
                    "waiting for ROS catalog metadata",
                    LogType.WARN,
                    throw_when_excep=True)
            warn_counter += 1
            time.sleep(0.05)

        return False

    def _init_catalog_client(self):

        remapped_catalog_ns = self._remap_namespace(self._namespace)
        self._catalog_client = ClientFactory(
            basename=self._CATALOG_BASENAME,
            namespace=remapped_catalog_ns,
            verbose=self._verbose,
            vlevel=self._vlevel,
            dtype=dtype.Int,
            safe=True,
        )
        self._catalog_client.attach()

        n_rows = self._catalog_client.getNRows()
        n_cols = self._catalog_client.getNCols()
        self._catalog_buffer = np.zeros((n_rows, n_cols), dtype=np.int32)

    def _decode_string_tensor(self, np_data):

        decoded = []

        for col_idx in range(np_data.shape[1]):
            raw = bytearray()
            terminated = False

            for row_idx in range(np_data.shape[0]):
                value = int(np_data[row_idx, col_idx])
                for byte_idx in range(4):
                    byte = (value >> (8 * byte_idx)) & 0xFF
                    if byte == 0:
                        terminated = True
                        break
                    raw.append(byte)
                if terminated:
                    break

            decoded.append(raw.decode("utf-8", errors="ignore"))

        return decoded

    def _parse_catalog_specs(self, entries):

        stream_specs = []
        seen = set()

        for entry in entries:
            if entry is None:
                continue

            clean = entry.strip()
            if clean == "":
                continue

            if "|" not in clean:
                Journal.log(self.__class__.__name__,
                    "_parse_catalog_specs",
                    f"ignoring malformed catalog entry '{clean}'",
                    LogType.WARN,
                    throw_when_excep=True)
                continue

            namespace, basename = clean.split("|", 1)
            if basename == "" or basename == self._CATALOG_BASENAME:
                continue

            key = (basename, namespace)
            if key in seen:
                continue
            seen.add(key)

            stream_specs.append((basename, namespace))

        return stream_specs

    def _read_catalog_once(self):

        if self._catalog_client is None or self._catalog_buffer is None:
            return None

        read_ok = self._catalog_client.read(self._catalog_buffer, 0, 0)
        if not read_ok:
            return None

        entries = self._decode_string_tensor(self._catalog_buffer)

        return self._parse_catalog_specs(entries)

    def _ensure_bridges_from_catalog(self, stream_specs):

        for basename, namespace in stream_specs:
            key = (basename, namespace)
            if key in self._bridges:
                continue

            kwargs = dict(
                basename=basename,
                namespace=namespace,
                queue_size=self._queue_size,
                ros_backend=self._backend,
                verbose=self._verbose,
                vlevel=self._vlevel,
                force_reconnection=self._force_reconnection,
                remap_ns=self._remap_namespace(namespace),
            )
            if self._backend == "ros2":
                kwargs["node"] = self._node

            self._bridges[key] = {
                "bridge": FromRos(**kwargs),
                "ready": False,
            }

            Journal.log(self.__class__.__name__,
                "_ensure_bridges_from_catalog",
                f"discovered ROS stream {namespace}/{basename}",
                LogType.INFO,
                throw_when_excep=True)

    def _refresh_catalog(self):

        stream_specs = self._read_catalog_once()
        if stream_specs is None:
            return False

        self._ensure_bridges_from_catalog(stream_specs)

        return True

    def _update_discovered_bridges(self):

        pending_ids = []

        for (basename, namespace), state in self._bridges.items():
            bridge = state["bridge"]

            if not state["ready"]:
                if bridge.run():
                    state["ready"] = True
                    Journal.log(self.__class__.__name__,
                        "_update_discovered_bridges",
                        f"bridge ready for {namespace}/{basename}",
                        LogType.INFO,
                        throw_when_excep=True)
                else:
                    pending_ids.append(f"{basename}@{namespace}")
                    continue

            bridge.update()

        if len(pending_ids) > 0:
            if self._warn_counter % 20 == 0:
                Journal.log(self.__class__.__name__,
                    "_update_discovered_bridges",
                    f"waiting for ROS metadata on {len(pending_ids)} bridge(s): {', '.join(pending_ids)}",
                    LogType.WARN,
                    throw_when_excep=True)
            self._warn_counter += 1

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_backend()
        self._init_catalog_bridge()

        if not self._wait_catalog_bridge_ready():
            Journal.log(self.__class__.__name__,
                "run",
                "failed to initialize catalog bridge",
                LogType.WARN,
                throw_when_excep=True)
            self.close()
            return

        self._init_catalog_client()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = f"starting ROS-to-shared-memory bridge with update dt {self._dt} s" + \
            f" and namespace {self._namespace} ({self._backend}), remapped to {self._remap_ns}"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep=True)

        while self._is_running and self._backend_alive():
            try:
                start_time = time.perf_counter()
                self._update()
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep_ns = int((self._dt - elapsed_time) * 1e9)
                if time_to_sleep_ns < 0:
                    Journal.log(self.__class__.__name__,
                        "run",
                        f"Could not match desired update dt of {self._dt} s. Elapsed {elapsed_time} s.",
                        LogType.WARN,
                        throw_when_excep=True)
                else:
                    PerfSleep.thread_sleep(time_to_sleep_ns)
            except (KeyboardInterrupt, SystemExit):
                break

        self.close()

    def _update(self):

        self._spin_backend_once()

        if self._catalog_bridge is not None:
            self._catalog_bridge.update()

        self._refresh_catalog()
        self._update_discovered_bridges()

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and self._catalog_bridge is None:
            return

        self._is_running = False

        for state in self._bridges.values():
            try:
                state["bridge"].close()
            except Exception:
                pass

        self._bridges = {}

        if self._catalog_bridge is not None:
            try:
                self._catalog_bridge.close()
            except Exception:
                pass
            self._catalog_bridge = None

        if self._catalog_client is not None:
            try:
                self._catalog_client.close()
            except Exception:
                pass
            self._catalog_client = None

        self._catalog_buffer = None

        self._shutdown_backend()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ROS to shared-memory bridge")
    parser.add_argument('--cores', nargs='+', type=str,
        help='CPU cores to set affinity (examples: "2 3 4", "2-5", "2,4,6")')
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--remap_ns', type=str, default=None,
        help='Namespace to be used for remapping when creating shared memory servers')
    parser.add_argument('--add_training_data', action='store_true',
        help='Reserved, kept for CLI compatibility')

    args = parser.parse_args()

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    backend = "ros2" if args.ros2 else "ros1"

    bridge = RosToSharedMemBridge(namespace=args.ns,
                    backend=backend,
                    add_training_data=args.add_training_data,
                    remap_ns=args.remap_ns)

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
