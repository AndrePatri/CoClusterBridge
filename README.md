# CoClusterBridge
If you have a beautiful CPU-based controller (developed, for example, with Trajectory Optimization tools like [horizon-casadi](https://github.com/ADVRHumanoids/horizon)) and want to implement a *Learning-Based Receding Horizon Control* approach without having to rewrite your whole controller on GPU, then this might be the right tool for you. 

CoClusterBridge is a tool for bridging parallel simulations (typically GPU-based), with a cluster of CPU-based (receding-horizon) controllers and properly handling controllers synchronization and triggering.