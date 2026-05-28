from EigenIPC.PyEigenIPC import VLevel, LogType, Journal
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.to_zmq import ToZmq
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import default_endpoint
from mpc_hive.utilities.timing import high_resolution_sleep_ns

from mpc_hive.utilities.shared_data.rhc_data import RobotState, RhcRefs, RhcCmds, RhcStatus, RhcPred, RhcPredDelta
from mpc_hive.utilities.shared_data.cluster_profiling import RhcProfiling
from mpc_hive.utilities.shared_data.rhc_data import RhcInternal
from mpc_hive.utilities.shared_data.sim_data import SharedEnvInfo
from mpc_hive.utilities.shared_data.jnt_imp_control import JntImpCntrlData
from mpc_hive.utilities.shared_data.abstractions import flatten_shared_mem

import argparse
import time

from mpc_hive.utilities.sysutils import set_process_affinity, parse_env_slice


class SharedMemToZmqBridge:

    _CATALOG_BASENAME = "ZmqBridgeCatalog"

    def __init__(self,
            namespace: str,
            add_rhc_internal: bool = False,
            env_idx: int = None,
            env_count: int = 1,
            verbose: bool = True,
            vlevel: VLevel = VLevel.V2,
            queue_size: int = 1,
            conflate: bool = True,
            bind: bool = True,
            bind_ip: str = "0.0.0.0",
            port_base: int = 20000,
            port_span: int = 40000,
            timing_window: int = 200,
            string_stream_period: float = 1.0,
            string_stream_once: bool = False,
            drop_if_busy: bool = True):

        self._namespace = namespace
        self._add_rhc_internal = add_rhc_internal
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
        self._timing_window = max(1, int(timing_window))
        self._string_stream_period = max(0.0, float(string_stream_period))
        self._string_stream_once = bool(string_stream_once)
        self._drop_if_busy = bool(drop_if_busy)

        self._bridges = []
        self._bridge_meta = []
        self._clients = []
        self._shared_mems = []
        self._rhc_internal_shared_mems = set()
        self._unsliced_shared_mems = set()
        self._string_shared_mems = set()

        self._catalog_server = None
        self._catalog_client = None
        self._catalog_bridge = None
        self._catalog_strings = []

        self._dt = 0.05
        self._is_running = False
        self._timing_samples = 0
        self._timing_overruns = 0
        self._timing_elapsed_sum = 0.0
        self._timing_elapsed_max = 0.0
        self._timing_overrun_sum = 0.0
        self._timing_overrun_max = 0.0
        self._catalog_publish_period = 1.0
        self._last_catalog_publish_t = 0.0

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

    def _as_meta_list(self, meta):

        if meta is None:
            return []

        if isinstance(meta, (list, tuple)):
            return list(flatten_shared_mem(meta))

        return [meta]

    def _apply_client_shm_meta(self,
            client,
            client_mems):

        if len(client_mems) == 0:
            return

        if not hasattr(client, "get_shm_type") or not callable(getattr(client, "get_shm_type")):
            Journal.log(self.__class__.__name__,
                "_apply_client_shm_meta",
                f"Client {client.__class__.__name__} does not expose get_shm_type().",
                LogType.EXCEP,
                throw_when_excep=True)

        if not hasattr(client, "get_shm_sliceable") or not callable(getattr(client, "get_shm_sliceable")):
            Journal.log(self.__class__.__name__,
                "_apply_client_shm_meta",
                f"Client {client.__class__.__name__} does not expose get_shm_sliceable().",
                LogType.EXCEP,
                throw_when_excep=True)

        client_types = self._as_meta_list(client.get_shm_type())
        if len(client_types) != len(client_mems):
            Journal.log(self.__class__.__name__,
                "_apply_client_shm_meta",
                f"Client {client.__class__.__name__} returned {len(client_types)} entries in get_shm_type "
                f"for {len(client_mems)} shared memories.",
                LogType.EXCEP,
                throw_when_excep=True)

        client_sliceable = self._as_meta_list(client.get_shm_sliceable())
        if len(client_sliceable) != len(client_mems):
            Journal.log(self.__class__.__name__,
                "_apply_client_shm_meta",
                f"Client {client.__class__.__name__} returned {len(client_sliceable)} entries in get_shm_sliceable "
                f"for {len(client_mems)} shared memories.",
                LogType.EXCEP,
                throw_when_excep=True)

        for idx, shm_type in enumerate(client_types):
            if shm_type == "str_list":
                self._unsliced_shared_mems.add(id(client_mems[idx]))
                self._string_shared_mems.add(id(client_mems[idx]))

        for idx, is_sliceable in enumerate(client_sliceable):
            if not bool(is_sliceable):
                self._unsliced_shared_mems.add(id(client_mems[idx]))

    def _run_clients(self):

        self._shared_mems = []
        self._rhc_internal_shared_mems = set()
        self._unsliced_shared_mems = set()
        self._string_shared_mems = set()
        for client in self._clients:
            client.run()
            if not client.is_running():
                client_name = client.__class__.__name__
                Journal.log(self.__class__.__name__,
                    "_run_clients",
                    f"Client {client_name} failed to start",
                    LogType.ERROR,
                    throw_when_excep=True)
            client_mems = self._as_mem_list(client.get_shared_mem())
            self._shared_mems.extend(client_mems)
            self._apply_client_shm_meta(client, client_mems)

        self._run_rhc_internal_clients()

    def _infer_cluster_size(self):

        for client in self._clients:
            if isinstance(client, RobotState):
                return client.n_robots()

        for client in self._clients:
            if isinstance(client, RhcStatus):
                return int(client.trigger.n_rows)

        Journal.log(self.__class__.__name__,
            "_infer_cluster_size",
            "Could not infer cluster size from existing clients.",
            LogType.EXCEP,
            throw_when_excep=True)

    def _selected_rhc_indices(self,
            cluster_size: int):

        if self._env_idx is None:
            return list(range(cluster_size))

        if self._env_idx >= cluster_size:
            Journal.log(self.__class__.__name__,
                "_selected_rhc_indices",
                f"env_idx={self._env_idx} is out of bounds for cluster size {cluster_size}.",
                LogType.EXCEP,
                throw_when_excep=True)

        end_idx = min(self._env_idx + self._env_count, cluster_size)
        return list(range(self._env_idx, end_idx))

    def _run_rhc_internal_clients(self):

        if not self._add_rhc_internal:
            return

        cluster_size = self._infer_cluster_size()
        selected_indices = self._selected_rhc_indices(cluster_size)

        for rhc_idx in selected_indices:
            rhc_internal_client = RhcInternal(
                config=RhcInternal.Config(is_server=False),
                namespace=self._namespace,
                rhc_index=rhc_idx,
                safe=False,
                verbose=self._verbose,
                vlevel=self._vlevel,
            )
            rhc_internal_client.run()

            if not rhc_internal_client.is_running():
                Journal.log(self.__class__.__name__,
                    "_run_rhc_internal_clients",
                    f"RhcInternal client for index {rhc_idx} failed to start.",
                    LogType.ERROR,
                    throw_when_excep=True)

            self._clients.append(rhc_internal_client)
            rhc_mems = self._as_mem_list(rhc_internal_client.get_shared_mem())
            self._shared_mems.extend(rhc_mems)
            self._apply_client_shm_meta(rhc_internal_client, rhc_mems)
            for mem in rhc_mems:
                self._rhc_internal_shared_mems.add(id(mem))

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

    def _publish_catalog(self,
            force: bool = False):

        if self._catalog_server is None or self._catalog_bridge is None:
            return

        now = time.perf_counter()
        if not force and (now - self._last_catalog_publish_t) < self._catalog_publish_period:
            return

        payload = self._catalog_strings if len(self._catalog_strings) > 0 else [""]

        self._catalog_server.write_vec(payload, 0)
        self._catalog_bridge.update(retry=False)
        self._last_catalog_publish_t = now

    def _accumulate_timing(self,
            elapsed_time: float):

        overrun = max(0.0, elapsed_time - self._dt)
        self._timing_samples += 1
        self._timing_elapsed_sum += elapsed_time
        if elapsed_time > self._timing_elapsed_max:
            self._timing_elapsed_max = elapsed_time

        if overrun > 0.0:
            self._timing_overruns += 1
            self._timing_overrun_sum += overrun
            if overrun > self._timing_overrun_max:
                self._timing_overrun_max = overrun

        if self._timing_samples >= self._timing_window:
            if self._timing_overruns > 0:
                avg_elapsed = self._timing_elapsed_sum / self._timing_samples
                avg_overrun = self._timing_overrun_sum / self._timing_overruns
                Journal.log(self.__class__.__name__,
                    "run",
                    f"Timing window {self._timing_samples} samples: "
                    f"overruns {self._timing_overruns}/{self._timing_samples}, "
                    f"avg_elapsed={avg_elapsed:.6f}s, max_elapsed={self._timing_elapsed_max:.6f}s, "
                    f"avg_overrun={avg_overrun:.6f}s, max_overrun={self._timing_overrun_max:.6f}s.",
                    LogType.WARN,
                    throw_when_excep=True)
            self._timing_samples = 0
            self._timing_overruns = 0
            self._timing_elapsed_sum = 0.0
            self._timing_elapsed_max = 0.0
            self._timing_overrun_sum = 0.0
            self._timing_overrun_max = 0.0

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
        self._bridge_meta = []
        for shared_mem in self._shared_mems:
            is_rhc_internal_stream = id(shared_mem) in self._rhc_internal_shared_mems
            is_unsliced_stream = id(shared_mem) in self._unsliced_shared_mems
            is_string_stream = id(shared_mem) in self._string_shared_mems
            source_row_index = None if (is_rhc_internal_stream or is_unsliced_stream) else self._env_idx
            source_n_rows = 1 if (is_rhc_internal_stream or is_unsliced_stream) else self._env_count

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
                drop_if_busy=self._drop_if_busy,
                source_row_index=source_row_index,
                source_n_rows=source_n_rows,
            )
            bridge.run()
            self._bridges.append(bridge)
            self._bridge_meta.append({
                "is_string": is_string_stream,
                "next_publish_t": 0.0,
                "published_once": False,
            })

            Journal.log(self.__class__.__name__,
                "_init_to_zmq_bridges",
                f"publishing {shared_mem.getNamespace()}/{shared_mem.getBasename()} on {endpoint} "
                f"(env_idx={source_row_index}, env_count={source_n_rows})",
                LogType.INFO,
                throw_when_excep=True)

    def run(self, dt: float = 0.05):

        self._dt = dt
        self._catalog_publish_period = max(0.5, 20.0 * self._dt)
        self._last_catalog_publish_t = 0.0

        self._init_clients()
        self._run_clients()
        self._init_to_zmq_bridges()
        self._init_catalog_bridge()
        self._publish_catalog(force=True)

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
            start_time = time.perf_counter()
            self._update()
            elapsed_time = time.perf_counter() - start_time
            self._accumulate_timing(elapsed_time)
            time_to_sleep_ns = int((self._dt - elapsed_time) * 1e9)
            if time_to_sleep_ns >= 0:
                high_resolution_sleep_ns(time_to_sleep_ns)

        self.close()

    def _update(self):

        self._publish_catalog()

        now = time.perf_counter()
        for idx, bridge in enumerate(self._bridges):
            meta = self._bridge_meta[idx]
            if meta["is_string"]:
                if self._string_stream_once and meta["published_once"]:
                    continue
                if now < meta["next_publish_t"]:
                    continue

            published = bridge.update(retry=False)

            if meta["is_string"] and published:
                meta["published_once"] = True
                if not self._string_stream_once:
                    meta["next_publish_t"] = now + self._string_stream_period

    def close(self):

        if not self._is_running and len(self._bridges) == 0 and len(self._clients) == 0:
            return

        self._is_running = False
        self._close_catalog()
        self._close_bridges()
        self._close_clients()
        self._bridges = []
        self._bridge_meta = []
        self._clients = []
        self._shared_mems = []
        self._rhc_internal_shared_mems = set()
        self._unsliced_shared_mems = set()
        self._string_shared_mems = set()


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
    parser.add_argument('--add_rhc_internal', action='store_true',
        help='Publish per-controller RhcInternal streams')
    parser.add_argument('--env_idx', type=str, default=None,
        help='Optional env index or inclusive range (examples: "67", "67-75")')
    parser.add_argument('--timing_window', type=int, default=200,
        help='Number of loop samples used to aggregate dt violation warnings')
    parser.add_argument('--string_stream_period', type=float, default=1.0,
        help='Publish period [s] for str_list streams (row/col names)')
    parser.add_argument('--string_stream_once', action='store_true',
        help='Publish str_list streams only once after startup')
    parser.add_argument('--no_drop_if_busy', action='store_true',
        help='Disable non-blocking PUB send (bridge may block when socket is busy)')

    args = parser.parse_args()

    if args.cores:
        selected = set_process_affinity(args.cores)
        print(f"Set CPU affinity to cores: {selected}")

    env_start, env_count = parse_env_slice(args.env_idx)

    bridge = SharedMemToZmqBridge(
        namespace=args.ns,
        add_training_data=args.add_training_data,
        add_rhc_internal=args.add_rhc_internal,
        env_idx=env_start,
        env_count=env_count,
        queue_size=args.queue_size,
        conflate=not args.no_conflate,
        bind_ip=args.bind_ip,
        port_base=args.port_base,
        port_span=args.port_span,
        timing_window=args.timing_window,
        string_stream_period=args.string_stream_period,
        string_stream_once=args.string_stream_once,
        drop_if_busy=not args.no_drop_if_busy,
    )

    try:
        bridge.run(dt=args.dt)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
