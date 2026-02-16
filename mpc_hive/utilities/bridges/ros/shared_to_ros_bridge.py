from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.to_ros import ToRos
from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcRefs, RhcCmds, RhcStatus
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling

import argparse
import time

from mpc_hive.utilities.sysutils import set_process_affinity, parse_env_slice

class SharedMemToRosBridge:

    _CATALOG_BASENAME = "RosBridgeCatalog"

    def __init__(self,
            namespace: str,
            backend: str = "ros2",
            add_training_data: bool = False,
            env_idx: int = None,
            env_count: int = 1,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1):

        self._namespace = namespace
        self._backend = backend
        self._add_training_data = add_training_data
        self._env_idx = env_idx
        self._env_count = env_count
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size

        self._bridges = []
        self._clients = []
        self._shared_mems = []

        self._catalog_server = None
        self._catalog_client = None
        self._catalog_bridge = None
        self._catalog_strings = []

        self._dt = 0.05
        self._is_running = False
        self._node = None

        self._check_backend()

        if self._env_idx is not None and self._env_idx < 0:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Invalid env_idx {self._env_idx}. It must be >= 0.",
                LogType.EXCEP,
                throw_when_excep=True)

        if self._env_count < 1:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Invalid env_count {self._env_count}. It must be >= 1.",
                LogType.EXCEP,
                throw_when_excep=True)

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

    def _build_extra_clients(self):

        # Hook for derived packages (e.g. AugMPC training streams).
        return []

    def _init_clients(self):

        self._clients = [
            RhcStatus(namespace=self._namespace,
                is_server=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RobotState(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcRefs(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcCmds(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcProfiling(name=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
        ]

        self._clients.extend(self._build_extra_clients())

    def _as_mem_list(self, shared_mem):

        if shared_mem is None:
            return []
        if isinstance(shared_mem, (list, tuple)):
            flat = []
            for item in shared_mem:
                flat.extend(self._as_mem_list(item))
            return flat
        return [shared_mem]

    def _deduplicate_shared_mems(self):

        unique = {}
        for shared_mem in self._shared_mems:
            key = (shared_mem.getNamespace(), shared_mem.getBasename())
            if key not in unique:
                unique[key] = shared_mem
        self._shared_mems = list(unique.values())

    def _run_clients(self):

        self._shared_mems = []
        for client in self._clients:
            client.run()
            self._shared_mems.extend(self._as_mem_list(client.get_shared_mem()))

        self._deduplicate_shared_mems()

    def _close_clients(self):

        for client in self._clients:
            try:
                client.close()
            except Exception:
                pass

    def _close_bridges(self):

        for bridge in self._bridges:
            try:
                bridge.close()
            except Exception:
                pass

    def _init_backend(self):

        if self._backend == "ros1":
            import rospy
            rospy.init_node("SharedMem2RosBridge_" + self._namespace)
        elif self._backend == "ros2":
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("SharedMem2RosBridge_" + self._namespace)

    def _init_to_ros_bridges(self):

        self._bridges = []
        for shared_mem in self._shared_mems:
            if self._backend == "ros1":
                bridge = ToRos(client=shared_mem,
                    queue_size=self._queue_size,
                    ros_backend=self._backend,
                    source_row_index=self._env_idx,
                    source_n_rows=self._env_count)
            else:
                bridge = ToRos(client=shared_mem,
                    queue_size=self._queue_size,
                    ros_backend=self._backend,
                    node=self._node,
                    source_row_index=self._env_idx,
                    source_n_rows=self._env_count)
            bridge.run()
            self._bridges.append(bridge)

    def _collect_stream_specs(self):

        stream_specs = []
        for shared_mem in self._shared_mems:
            stream_specs.append((shared_mem.getNamespace(), shared_mem.getBasename()))

        stream_specs.sort(key=lambda item: (item[0], item[1]))

        return stream_specs

    def _init_catalog_bridge(self):

        stream_specs = self._collect_stream_specs()
        self._catalog_strings = [f"{namespace}|{basename}" for namespace, basename in stream_specs]

        catalog_length = max(1, len(self._catalog_strings))

        self._catalog_server = StringTensorServer(
            length=catalog_length,
            basename=self._CATALOG_BASENAME,
            name_space=self._namespace,
            verbose=self._verbose,
            vlevel=self._vlevel,
            force_reconnection=True,
            safe=True,
        )
        self._catalog_server.run()

        self._catalog_client = StringTensorClient(
            basename=self._CATALOG_BASENAME,
            name_space=self._namespace,
            verbose=self._verbose,
            vlevel=self._vlevel,
            safe=True,
        )
        self._catalog_client.run()

        kwargs = dict(
            client=self._catalog_client,
            queue_size=self._queue_size,
            ros_backend=self._backend,
        )
        if self._backend == "ros2":
            kwargs["node"] = self._node

        self._catalog_bridge = ToRos(**kwargs)
        self._catalog_bridge.run()

        Journal.log(self.__class__.__name__,
            "_init_catalog_bridge",
            f"publishing ROS catalog with {len(self._catalog_strings)} stream(s)",
            LogType.INFO,
            throw_when_excep=True)

    def _publish_catalog(self):

        if self._catalog_server is None or self._catalog_bridge is None:
            return

        payload = self._catalog_strings if len(self._catalog_strings) > 0 else [""]
        self._catalog_server.write_vec(payload, 0)
        self._catalog_bridge.update()

    def _close_catalog(self):

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

        if self._catalog_server is not None:
            try:
                self._catalog_server.close()
            except Exception:
                pass
            self._catalog_server = None

        self._catalog_strings = []

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_backend()
        self._init_clients()
        self._run_clients()
        self._init_to_ros_bridges()
        self._init_catalog_bridge()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = f"starting shared memory-to-ROS bridge with update dt {self._dt} s" + \
            f" and namespace {self._namespace} ({self._backend})"
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
                time_to_sleep = self._dt - elapsed_time
                if time_to_sleep < 0:
                    Journal.log(self.__class__.__name__,
                        "run",
                        f"Could not match desired update dt of {self._dt} s. Elapsed {elapsed_time} s.",
                        LogType.WARN,
                        throw_when_excep=True)
                else:
                    time.sleep(time_to_sleep)
            except (KeyboardInterrupt, SystemExit):
                break

        self.close()

    def _update(self):

        self._publish_catalog()

        for bridge in self._bridges:
            bridge.update()

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._clients) == 0:
            return

        self._is_running = False
        self._close_catalog()
        self._close_bridges()
        self._close_clients()
        self._shutdown_backend()
        self._bridges = []
        self._clients = []
        self._shared_mems = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Shared-memory to ROS bridge")
    parser.add_argument('--cores', nargs='+', type=str,
        help='CPU cores to set affinity (examples: "2 3 4", "2-5", "2,4,6")')
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--add_training_data', action='store_true',
        help='Reserved for derived bridge implementations')
    parser.add_argument('--env_idx', type=str, default=None,
        help='Optional env index or inclusive range (examples: "67", "67-75")')
    args = parser.parse_args()

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    env_start, env_count = parse_env_slice(args.env_idx)

    backend = "ros2" if args.ros2 else "ros1"

    bridge = SharedMemToRosBridge(namespace=args.ns,
                    backend=backend,
                    add_training_data=args.add_training_data,
                    env_idx=env_start,
                    env_count=env_count)

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
