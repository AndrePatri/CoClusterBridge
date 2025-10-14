<h2 align="center" style="text-decoration: none;"> <img src="https://img.shields.io/badge/License-GPLv2-purple.svg" alt="License">

![icon.svg](docs/images/logo.svg)

</h2>
If you have already developed a beautiful MPC controller on CPU and want to embed it within a RL pipeline while keeping the MPCs on CPU, then this might be the right tool for you. 

MPCHive is a tool for bridging parallel simulations (typically GPU-based), with a cluster (or "hive") of CPU-based MPCs, while properly handling controllers' synchronization and triggering (full CPU operation is also supported).
One direct application for MPCHive is for developing RL-augmented MPC policies as done in [IBRIDO](https://github.com/AndrePatri/AugMPC), but it can also be used in a standalone fashion for swarm robotics (e.g. when having a fleet of MPC-controlled robots) or for massive MPC benchmarking.
