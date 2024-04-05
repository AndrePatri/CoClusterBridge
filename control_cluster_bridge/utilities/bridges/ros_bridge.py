from SharsorIPCpp.PySharsor.extensions.ros_bridge.to_ros import *
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import *
from SharsorIPCpp.PySharsorIPC import *

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal

import argparse
import time 

from perf_sleep.pyperfsleep import PerfSleep

class Sharsor2RosBridge():

    def __init__(self,
            namespace: str,
            backend: str = "ros2"):

        self._namespace = namespace
        self._backend = backend

        self._bridges = []
        self._clients = []

        self._init_clients()

        self._dt = 0.05

        self._is_running = False

    def _init_clients(self):
        self._clients.append(RobotState(namespace=self._namespace, 
                                is_server=False, 
                                safe=False, 
                                verbose=True, 
                                vlevel=VLevel.V1))
                
    def _run_clients(self):
        for client in self._clients:
            client.run()

    def _close_clients(self):
        for client in self._clients:
            client.close()

    def _init_toROS_bridges(self):

        if self._backend == "ros1":
            import rospy
            node = rospy.init_node("Sharsor2RosBridge_" + self._namespace)
            for client in self._clients:
                self._bridges.append(ToRos(client=client.get_shared_mem(),
                    queue_size = 1,
                    ros_backend = self._backend))
            
        elif self._backend == "ros2":
            import rclpy
            rclpy.init()
            node = rclpy.create_node(self._namespace)
            for client in self._clients:

                shared_mems = client.get_shared_mem() # potentially a single Server/Client, a List or None

                self._bridges.append(ToRos(client=shared_mems,
                    queue_size = 1,
                    ros_backend = self._backend),
                    node=node)
        else:
            Journal.log(self.__class__.__name__,
                "_init_toROS_bridges",
                f"backend {self._backend} not supported!",
                LogType.EXCEP,
                throw_when_excep = True)

    def run(self, dt: float = 0.05):

        self._dt = dt

        self._run_clients()

        self._init_toROS_bridges()

        self._is_running = True

        self._run()

    def _run(self):

        info = f": starting sharsor-to-ROS bridge with update dt {self._dt} s" + \
            f" with namespace {self._namespace}"
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        start_time = 0.0
        elapsed_time = 0.0
        time_to_sleep_ns = 0

        while self._is_running:
            try:
                start_time = time.perf_counter() 
                self._update()
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep_ns = int((self._dt - elapsed_time) * 1000000000) # [ns]
                if time_to_sleep_ns < 0:
                    warning = f": Could not match desired update dt of {self._dt} s. " + \
                        f"Elapsed time to update {elapsed_time}."
                    Journal.log(self.__class__.__name__,
                        "run",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)
                else:
                    PerfSleep.thread_sleep(time_to_sleep_ns) 
                continue
            except KeyboardInterrupt:
                self.close()

    def _update(self):

        aaa = 1

    def close(self):

        self._close_clients()

        self._is_running = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Multi Robot Visualizer")
    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--ros2', action='store_true', help='Enable ROS 2 mode')
    parser.add_argument('--dt', type=float, default=0.01, help='Update interval in seconds, default is 0.01')

    args = parser.parse_args()
    
    backend = "ros2" if args.ros2 else "ros1"

    if args.ns is None:
        Journal.log("ros_bridge.py",
                "ros_bridge",
                "no --ns argument provided!",
                LogType.EXCEP,
                throw_when_excep = True)
    bridge = Sharsor2RosBridge(namespace=args.ns,
                    backend=backend)

    bridge.run()

    bridge.close()