# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of CoClusterBridge and distributed under the General Public License version 2 license.
# 
# CoClusterBridge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# CoClusterBridge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CoClusterBridge.  If not, see <http://www.gnu.org/licenses/>.
# 
import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        
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
