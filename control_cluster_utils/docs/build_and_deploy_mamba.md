### Steps to build this package and deploy it on conda

- go to the root of the package, where the `meta.yaml` is stored
- activate a suitable mamba environment
- run `mamba build .` (this might take a while to complete)
- open a terminal and run `anaconda login` (`pip install anaconda-client`)
- cd to the location where the .tar.bz2 was generated
- run `anaconda upload -c AndPatr controlclusterutils-1.0-py38.tar.bz2` to upload to the channel 
- to install `mamba install -c "andpatr/label/AndPatr" control_cluster_utils`
