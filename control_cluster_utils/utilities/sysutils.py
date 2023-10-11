# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of ControlClusterUtils and distributed under the General Public License version 2 license.
# 
# ControlClusterUtils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# ControlClusterUtils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with ControlClusterUtils.  If not, see <http://www.gnu.org/licenses/>.
# 
import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

        self.PIPES_CONFIGPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'config', 
                                            'pipes', 
                                            'pipes_config.yaml')
        
        self.SHARED_MEM_CONFIGPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'config', 
                                            'shared_mem', 
                                            'shared_mem_config.yaml')
        
        self.CONTROLLERS_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'controllers')
        
        self.CLUSTER_SRV_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'cluster_server')

        self.CLUSTER_CLT_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'cluster_client')

        self.UTILS_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'utilities')
        
        self.GUI_ICONS_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'docs', 
                                            'images', 
                                            'gui_icons')
