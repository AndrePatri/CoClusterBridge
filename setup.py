"""Installation script for the 'control_cluster_utils' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy",
    "torch", 
    "multiprocessing"
    ]

# Installation operation
setup(
    name="control_cluster_utils",
    author="AndPatr",
    version="0.0.1",
    description="",
    keywords=["cluster", "rhc", "controller"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
