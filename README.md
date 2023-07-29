# ControlClusterUtils

Control Cluster Utilities to spawn a number of controllers over multiple processes.
 
Used to bridge GPU-based simulators, e.g. [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), with come CPU-based controllers, developed, for example, with tools like [horizon-casadi](https://github.com/ADVRHumanoids/horizon) and still retain some resemblance of parallelism during training of RL agents.

The package is made of the following components:
- A `ControlClusterSrvr` object is in charge of loading and spawning a number of controllers over separate processes. The controllers must inherit and conform to the structure of a base `RHController` class. Controllers (assumed to be of the same type) are added to the server via the `add_controller` method and have to be created separately.
- A `ControlClusterClient` object represents the bridge between the controllers and the GPU-based simulation. 
- Each controller exposes a number of *named pipes* for receiving and send data to and from the `ControlClusterClient`
- Additional pipes are used for communication important initialization data between the `ControlClusterSrvr` and the `ControlClusterClient` object which, in general, live in separate processes. This might be useful feature to avoid library conflicts between the controllers and the simulator process.

