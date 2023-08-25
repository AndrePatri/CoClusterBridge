"""Installation script for the 'control_cluster_utils' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os
import sys

root_dir = os.path.dirname(os.path.realpath(__file__))

# Modify the version name based on Python version
python_version = ".".join(map(str, sys.version_info[:2]))  # Get major.minor Python version
package_name = 'control_cluster_utils'

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy",
    "torch", 
    "multiprocess"
    ]

# Installation operation
setup(
    name=package_name,
    author="AndrePatri",
    version="0.0.1-py" + python_version,
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
