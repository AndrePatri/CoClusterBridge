# ControlClusterUtils

Utilities to bridge parallel simulations (e.g. GPU-based simulators, e.g. [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html)), with a cluster of CPU-based controllers (developed, for example, with tools like [horizon-casadi](https://github.com/ADVRHumanoids/horizon)). 

The package is made of the following components:
- A `ControlClusterSrvr` object is in charge of loading and spawning a number of controllers over separate processes. Each controller must inherit and  to tconformhe structure of a base `RHController` class. Controllers are added to the server via the `add_controller` method.
- A `ControlClusterClient` object represents the bridge between the controllers and the parallel simulation (e.g. Omniverse Isaac Sim). 
- Each controller exposes a number of *named pipes* for receiving and sending data to and from the `ControlClusterClient`.
- Additional pipes are used for communicating important initialization data between the `ControlClusterSrvr` and the `ControlClusterClient` object which, in general, can live in separate processes. This feature might be useful to avoid library conflicts between the controllers and the simulator process (for example, currently Omniverse Isaac Sim must run in aits own Python3.7 environment with specific library versions).

