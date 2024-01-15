# CoClusterBridge
If you have a beautiful CPU-based controller (developed, for example, with Trajectory Optimization tools like [horizon-casadi](https://github.com/ADVRHumanoids/horizon)) and want to implement a *Learning-Based Receding Horizon Control* approach without having to rewrite your whole controller on GPU, then this might be the right tool for you. 
Ideally, one would want both the simulator and the controllers to live on GPU, but high-level tools for Trajectory Optimization are not available (yet).

CoClusterBridge is a tool for bridging parallel simulations (typically GPU-based), with a cluster of CPU-based (receding-horizon) controllers. 

<center><img src="control_cluster_bridge/docs/images/overview/architecture.png" alt="drawing" width="1000"/> </center>

The package is tailored to *Learning-Based Receding Horizon Control* approaches, in which classical Optimal Control and Trajectory Optimization meet Machine Learning. 

For instance, a possible usage for this package is in frameworks similar to the following one:
<center><img src="control_cluster_bridge/docs/images/overview/learning_based_rhc.png" alt="drawing" width="900"/> </center>

where a R.L. agent is coupled with a MPC controller. This allows to maintain performance, while guaranteeing constraints satisfaction and, hence, safety. 

This tool can be easily embedded into gyms (e.g [Gymnasium](https://gymnasium.farama.org/)) to perform the training phase of a R.L. agent. For minimal usage examples of this package with [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), have a look [here](https://github.com/AndrePatri/LRhcExamples).

As can be seen from the top picture, the package is made, at its core, of the following components:
- A `ControlClusterSrvr` object is in charge of loading and spawning a number of controllers over separate child processes. Each controller must inherit from a base `RHController` class. Controllers are added to the server via the `add_controller` method.
- A `ControlClusterClient` object represents the interface between the controllers and the parallel simulation environment (e.g. Omniverse Isaac Sim). 
- Data is shared and passed between *cluster server*, *cluster client* and the *control cluster* employing shared memory, for minimum latency (no need for serialization/deserialization and/or messages exchange) and maximum flexibility. 
The low-level implementation of the shared data mechanism is hosted in `utilities/shared_mem.py`. Specifically, the package uses [posix_ipc](https://github.com/osvenskan/posix_ipc) and [mmap](https://docs.python.org/3.7/library/mmap.html) to build shared memory clients and servers which create and manage (Torch) views of specific memory regions. 
- At its current version, the framework uses 3 main shared data structures built on top of the low-level memory mechanism: `RobotClusterState` for the state of the whole cluster (from the simulator), `RobotClusterCmd` for the commands computed by the controllers and `RhcClusterTaskRefs` for the run-time configurable parameters of the controllers. Each of these objects basically holds a big tensor and a series of views of it, which represent the data it holds (e.g. root position, orientation, etc...). Additionally, each of them also holds a *mirror* of their tensor on GPU. When necessary, the mirror is synched with its CPU counterpart or viceversa.
- When `ControlClusterClient`'s `solve` is called, the shared cluster state is synched with the one from the simulator (this might require a copy from GPU to CPU), all the controllers in the cluster run the solution of their associated control problem and fill a shared solution object with updated data. This final passage also implies an additional copy, this time from CPU to GPU, of the obtained control commands. By design, the client's `solve` will block until all controllers have returned. This way, the cluster is always synchronized with the simulator.

- Additionally, a debugging Qt5-based gui is also provided:

    <center><img src="control_cluster_bridge/docs/images/overview/debugger_gui.png" alt="drawing" width="600"/> </center>

    At its current state, the GUI has the following main features:
    - selection of which shared data to monitor (all or a subset of them).
    - arbitrary plot resizing and window collapsing
    - online window length resizing
    - online sample and update rate selection
    - for each window, line selection/deselection
    - pause/unpause of all or individual plots for better debugging
    - control cluster triggering
    - day/night mode

Some notes: 
- The package is also available through Anaconda at [control_cluster_bridge](https://anaconda.org/AndrePatri/control_cluster_bridge). `CoClusterBridge` is under active development, so its Anaconda version might not be always updated with the tip of this repo. For cutting-edge features, always refer to the source code hosted here.
- The reasons for using the third party library `posix_ipc` instead of the newest [multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) are twofold. First, `posix_ipc` is more flexible since it allows potentially communication with non-python applications. Second, and more importantly, `multiprocessing.shared_memory` is available only from Python 3.8 onwards and this might cause issues if interfacing with a simulator supporting earlier versions of Python (like IsaacSim 2022.2.1, which is only compatible with Python 3.7). Choosing `posix_ipc` thus enables maximum compatibility.

#### ToDo:

- [] Expose rhc status to GUI
- [] Test reset of single environment and single robot
- [] add check to match controller rate in sim with cluster rate
- [] Test rhc reset after failure, add reset and reset all buttons to reset a single or multiple rhc controllers (and corresponding robot)
- [] Move old shared structures relying to new SharsorIPCpp. Remove distinction between cluster and sinlge rhc shared structure and add child class for GPU interface. Expose at the cluster client level, aside from state, cmds also the rhc status structure and move rhc info like cost etc.. into rhc status object
- [] merge all data in the observation tensor for the agent, penalize rhc total cost and controller failure over episode.
- [] use PPO to optimaze the policy of the agent, save policy. Load it in separate script and make a parser which evaluates agents commands and forwards them over shared memory. Sinchronize with sim stepping (at a custom frequency) and test over a simple task of lifting the robot base to an height
- [] scale to complex tasks, integrate and improve new MPC, eventually go to stepping...
- [] test bound squashing (inequality expansion) for integration of inequalities into iLQR 
