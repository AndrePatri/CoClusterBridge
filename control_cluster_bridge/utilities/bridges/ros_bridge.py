from SharsorIPCpp.PySharsor.extensions.ros_bridge.to_ros import *
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import *

from SharsorIPCpp.PySharsorIPC import *

from control_cluster_bridge.utilities.data import RhcInternal
# from control_cluster_bridge.utilities.rhc_defs import RHCState

import os   

# Function to set CPU affinity
def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

order = 'C'

namespace = "kyon0"

config = RhcInternal.Config(is_server=False, 
                        enable_q= True, 
                        enable_v=False, 
                        enable_a=False, 
                        enable_a_dot=False, 
                        enable_f=False,
                        enable_f_dot=False, 
                        enable_eff=False
                        )

rhc_internal = RhcInternal(config=config, 
                        namespace=namespace,
                        rhc_index = 0,
                        is_server=False,
                        vlevel=VLevel.V2,
                        verbose=True)

rhc_internal.run()

class RosBridge():

    def __init__(self):

        self.
# update_dt = 0.005
# start_time = time.perf_counter() 
# start_time = 0.0
# elapsed_time = 0.0
# actual_loop_dt = 0.0

# time_to_sleep_ns = 0
# debug = False

# perf_timer = PerfSleep()

# namespace = 'Shared2RosBridge'

# ros_backend = "ros1" # ros1, ros2
# node = None
# bridge = None

# if ros_backend == "ros1":

#     import rospy

#     node = rospy.init_node(namespace)

#     bridge = ToRos(client=client.shared_mem,
#         queue_size = 1,
#         ros_backend = ros_backend)

# if ros_backend == "ros2":

#     import rclpy

#     rclpy.init()

#     node = rclpy.create_node(namespace)

#     bridge = ToRos(client=client.shared_mem,
#         queue_size=1,
#         ros_backend=ros_backend,
#         node=node)

# bridge.run()

# msg = f"Will try to run the bridge at {1/update_dt} Hz."
# Journal.log("test_to_ros.py",
#             "",
#             msg,
#             LogType.INFO,
#             throw_when_excep = True)

# try:

#     set_affinity([9])
    
#     while True:
        
#         if ros_backend == "ros1":
            
#             if rospy.is_shutdown():

#                 break

#         if ros_backend == "ros2":
            
#             if not rclpy.ok():

#                 break

#         start_time = time.perf_counter() 

#         # server.get_numpy_view()[:, :] = np.random.rand(server.n_rows, server.n_cols)

#         bridge.update()
        
#         # if ros_backend == "ros2":

#         #     rclpy.spin_once(node) # processes callbacks

#         elapsed_time = time.perf_counter() - start_time

#         time_to_sleep_ns = int((update_dt - elapsed_time) * 1e+9) # [ns]

#         if time_to_sleep_ns < 0:

#             warning = f"Could not match desired update dt of {update_dt} s. " + \
#                 f"Elapsed time to update {elapsed_time}."

#             Journal.log("test_to_ros.py",
#                         "",
#                         warning,
#                         LogType.WARN,
#                         throw_when_excep = True)

#         perf_timer.thread_sleep(time_to_sleep_ns) 

#         # loop_rate.sleep()

#         actual_loop_dt = time.perf_counter() - start_time

#         if debug:

#             print(f"Actual loop dt {actual_loop_dt} s.")

# except KeyboardInterrupt:
#     print("\nCtrl+C pressed. Exiting the loop.")