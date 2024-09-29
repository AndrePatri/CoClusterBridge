<h2 align="center" style="text-decoration: none;"> <img src="https://img.shields.io/badge/License-GPLv2-purple.svg" alt="License">

![icon.svg](docs/images/logo.svg)

</h2>
If you have a beautiful CPU-based controller developed, for example, with CPU-based Trajectory Optimization tools and want to train and deploy  a *Reinforcement Learning-based Model Predictive Control* policy without having to rewrite your whole controller on GPU, then this might be the right tool for you. 

CoClusterBridge is a tool for bridging parallel simulations (typically GPU-based), with a cluster of CPU-based (receding-horizon) controllers while properly handling controllers synchronization and triggering. It also supports full CPU operation.
For an example application, please have a look at [IBRIDO](https://github.com/AndrePatri/IBRIDO).
