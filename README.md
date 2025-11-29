<h2 align="center" style="text-decoration: none;"> <img src="https://img.shields.io/badge/License-GPLv2-purple.svg" alt="License">

![icon.svg](docs/images/logo_new.svg)

</h2>
If you have developed a MPC controller on CPU and want to do some learning on top of it, while keeping the controllers on CPU, then this might be the right tool for you. 

MPCHive was born as a tool to aid data-hungry RL-augmented MPC policies (e.g. [AugMPC](https://github.com/AndrePatri/AugMPC)), where efficient parallelization is crucial for better and faster learning.
It can also be used in a standalone fashion for tuning/designing MPCs, swarm robotics (e.g. fleets of MPC-controlled robots), massive MPC benchmarking, sampling-based control and more. Basically, any application which requires many MPCs running in parallel and reliable synchronization between them, if a fit for MPCHive.
