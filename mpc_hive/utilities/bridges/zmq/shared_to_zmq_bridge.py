from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.to_zmq import ToZmq
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import default_endpoint

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcRefs, RhcCmds, RhcStatus, RhcPred, RhcPredDelta
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling
from mpc_hive.utilities.shared_data.rhc_data import RhcInternal
from mpc_hive.utilities.shared_data.sim_data import SharedEnvInfo
from mpc_hive.utilities.shared_data.jnt_imp_control import JntImpCntrlData

import argparse
import time

from perf_sleep.pyperfsleep import PerfSleep
from mpc_hive.utilities.sysutils import set_process_affinity, parse_env_slice


class SharedMemToZmqBridge:

    _CATALOG_BASENAME = "ZmqBridgeCatalog"

    def __init__(self,
            namespace: str,
            add_training_data: bool = False,
            env_idx: int = None,
            env_count: int = 1,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            conflate: bool = True,
            bind: bool = True,
            bind_ip: str = "0.0.0.0",
            port_base: int = 20000,
            port_span: int = 40000):

        self._namespace = namespace
        self._add_training_data = add_training_data
        self._env_idx = env_idx
        self._env_count = env_count
        self._verbose = verbose
        self._vlevel = vlevel
        self._queue_size = queue_size
        self._conflate = conflate
        self._bind = bind
        self._bind_ip = bind_ip
        self._port_base = port_base
        self._port_span = port_span

        self._bridges = []
        self._clients = []
        self._shared_mems = []

        self._catalog_server = None
        self._catalog_client = None
        self._catalog_bridge = None
        self._catalog_strings = []

        self._dt = 0.05
        self._is_running = False

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

    def _build_extra_clients(self):

        # Hook for derived packages (e.g. AugMPC training streams).
        return []

    def _init_clients(self):

        # rhc_internal_config = RhcInternal.Config(is_server=False, 
        #                 enable_q=True)

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
            RhcCmds(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcRefs(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcProfiling(name=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            RhcPred(namespace=self._namespace,
                                is_server=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel),
            RhcPredDelta(namespace=self._namespace,
                                is_server=False, 
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel),
            SharedEnvInfo(namespace=self._namespace,
                is_server=False,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel),
            JntImpCntrlData(is_server = False, 
                namespace = self._namespace, 
                verbose = self._verbose, 
                vlevel = self._vlevel,
                    safe=False)
        ]

        self._clients.extend(self._build_extra_clients())

    def _as_mem_list(self, shared_mem):

        if shared_mem is None:
            return []

        if isinstance(shared_mem, (list, tuple)):
            flat_list = []
            for mem in shared_mem:
                flat_list.extend(self._as_mem_list(mem))
            return flat_list

        return [shared_mem]

    def _run_clients(self):

        self._shared_mems = []
        for client in self._clients:
            client.run()
            if not client.is_running():
                client_name = client.__class__.__name__
                Journal.log(self.__class__.__name__,
                    "_run_clients",
                    f"Client {client_name} failed to start",
                    LogType.ERROR,
                    throw_when_excep=True)
            self._shared_mems.extend(self._as_mem_list(client.get_shared_mem()))

    def _collect_stream_specs(self):

        unique = set()
        stream_specs = []

        for shared_mem in self._shared_mems:
            namespace = shared_mem.getNamespace()
            basename = shared_mem.getBasename()
            key = (namespace, basename)
            if key in unique:
                continue
            unique.add(key)
            stream_specs.append(key)

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

        catalog_endpoint = default_endpoint(
            namespace=self._namespace,
            basename=self._CATALOG_BASENAME,
            ip=self._bind_ip,
            port_base=self._port_base,
            port_span=self._port_span,
        )

        self._catalog_bridge = ToZmq(
            client=self._catalog_client,
            endpoint=catalog_endpoint,
            bind=self._bind,
            queue_size=self._queue_size,
            conflate=self._conflate,
        )
        self._catalog_bridge.run()

        Journal.log(self.__class__.__name__,
            "_init_catalog_bridge",
            f"publishing catalog on {catalog_endpoint} ({len(self._catalog_strings)} stream(s))",
            LogType.INFO,
            throw_when_excep=True)

    def _publish_catalog(self):

        if self._catalog_server is None or self._catalog_bridge is None:
            return

        payload = self._catalog_strings if len(self._catalog_strings) > 0 else [""]

        self._catalog_server.write_vec(payload, 0)
        self._catalog_bridge.update(retry=False)

    def _close_catalog(self):

        if self._catalog_bridge is not None:
            try:
                self._catalog_bridge.close()
            except Exception:
                pass
            self._catalog_bridge = None

        if self._catalog_server is not None:
            try:
                self._catalog_server.close()
            except Exception:
                pass
            self._catalog_server = None

        self._catalog_client = None
        self._catalog_strings = []

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

    def _init_to_zmq_bridges(self):

        self._bridges = []
        for shared_mem in self._shared_mems:
            endpoint = default_endpoint(
                namespace=shared_mem.getNamespace(),
                basename=shared_mem.getBasename(),
                ip=self._bind_ip,
                port_base=self._port_base,
                port_span=self._port_span,
            )

            bridge = ToZmq(
                client=shared_mem,
                endpoint=endpoint,
                bind=self._bind,
                queue_size=self._queue_size,
                conflate=self._conflate,
                source_row_index=self._env_idx,
                source_n_rows=self._env_count,
            )
            bridge.run()
            self._bridges.append(bridge)

            Journal.log(self.__class__.__name__,
                "_init_to_zmq_bridges",
                f"publishing {shared_mem.getNamespace()}/{shared_mem.getBasename()} on {endpoint} "
                f"(env_idx={self._env_idx}, env_count={self._env_count})",
                LogType.INFO,
                throw_when_excep=True)

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._init_clients()
        self._run_clients()
        self._init_to_zmq_bridges()
        self._init_catalog_bridge()

        self._is_running = True
        self._run_loop()

    def _run_loop(self):

        info = (
            f"starting shared memory-to-ZMQ bridge with update dt {self._dt} s "
            f"and namespace {self._namespace}"
        )
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep=True)

        while self._is_running:
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

        self._publish_catalog()

        for bridge in self._bridges:
            bridge.update(retry=False)

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._clients) == 0:
            return

        self._is_running = False
        self._close_catalog()
        self._close_bridges()
        self._close_clients()
        self._bridges = []
        self._clients = []
        self._shared_mems = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Shared-memory to ZMQ bridge")
    parser.add_argument('--cores', nargs='+', type=str,
        help='CPU cores to set affinity (examples: "2 3 4", "2-5", "2,4,6")')
    parser.add_argument('--ns', type=str, required=True,
        help='Namespace to be used for cluster shared memory')
    parser.add_argument('--dt', type=float, default=0.01,
        help='Update interval in seconds, default is 0.01')
    parser.add_argument('--bind_ip', type=str, default='0.0.0.0',
        help='IP/interface to bind publisher sockets on')
    parser.add_argument('--queue_size', type=int, default=1,
        help='ZMQ publisher queue size (HWM)')
    parser.add_argument('--no_conflate', action='store_true',
        help='Disable latest-only behavior')
    parser.add_argument('--port_base', type=int, default=20000,
        help='Base port used by deterministic endpoint mapping')
    parser.add_argument('--port_span', type=int, default=40000,
        help='Port span used by deterministic endpoint mapping')
    parser.add_argument('--add_training_data', action='store_true',
        help='Reserved for derived bridge implementations')
    parser.add_argument('--env_idx', type=str, default=None,
        help='Optional env index or inclusive range (examples: "67", "67-75")')

    args = parser.parse_args()

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    env_start, env_count = parse_env_slice(args.env_idx)

    bridge = SharedMemToZmqBridge(
        namespace=args.ns,
        add_training_data=args.add_training_data,
        env_idx=env_start,
        env_count=env_count,
        queue_size=args.queue_size,
        conflate=not args.no_conflate,
        bind_ip=args.bind_ip,
        port_base=args.port_base,
        port_span=args.port_span,
    )

    try:
        bridge.run(dt=args.dt)
    finally:
        bridge.close()
