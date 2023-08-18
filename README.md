# ControlClusterUtils

Utilities to bridge parallel simulations (e.g. GPU-based simulators, e.g. [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html)), with a cluster of CPU-based controllers (developed, for example, with tools like [horizon-casadi](https://github.com/ADVRHumanoids/horizon)). 

The package is made of the following components:
- A `ControlClusterSrvr` object is in charge of loading and spawning a number of controllers over separate processes. Each controller must inherit from a base `RHController` class. Controllers are added to the server via the `add_controller` method.
- A `ControlClusterClient` object represents the bridge between the controllers and the parallel simulation (e.g. Omniverse Isaac Sim). 
- When `ControlClusterClient`'s `solve` is called, the shared cluster state is synched with the one from the simulator and all the controllers in the cluster run the solution of their associated control problem and fill the shared solution with updated data. By design, the client's `solve` will block until all controllers have returned. This way, it's super easy to synchronize the cluster with the simulator.
- Data is shared between server, client and controllers employing shared memory, for minimum latency (no need for serialization/deserialization and/or messages exchange) and maximum flexibility. 
The low-level implementation of the shared data mechanism is hosted in `utilities/shared_mem.py`. At its core, the package uses [posix_ipc](https://github.com/osvenskan/posix_ipc) and [mmap](https://docs.python.org/3.7/library/mmap.html) to build shared memory clients and servers which create and manage views of specific memory regions. 
The reasons for using the third party library posix_ipc instead of the newest [multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) are twofold. First, posix_ipc is more flexible since it allows potentially communication with non-python application. Second, and more importantly, multiprocessing.shared_memory is available only from Python 3.8 onwards and this might cause issues if interfacing with a simulator supporting earlier versions of Python (like IsaacSim 2022.2.1). Choosing posix_ipc thus enables maximum compatibility.
- The package is also available through Anaconda at [control_cluster_utils](https://anaconda.org/AndrePatri/control_cluster_utils).

