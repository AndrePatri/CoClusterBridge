# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of MPCHive and distributed under the General Public License version 2 license.
# 
# MPCHive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# MPCHive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MPCHive.  If not, see <http://www.gnu.org/licenses/>.
# 
import os
from typing import Iterable, List


def parse_core_list(core_tokens: Iterable[str]) -> List[int]:

    if core_tokens is None:
        return []

    cores: List[int] = []
    seen = set()

    for token in core_tokens:
        if token is None:
            continue

        for chunk in str(token).split(","):
            item = chunk.strip()
            if item == "":
                continue

            if "-" in item:
                parts = item.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Invalid core range '{item}'")
                start = int(parts[0])
                end = int(parts[1])
                if start < 0 or end < 0 or end < start:
                    raise ValueError(f"Invalid core range '{item}'")

                for core in range(start, end + 1):
                    if core not in seen:
                        seen.add(core)
                        cores.append(core)
            else:
                core = int(item)
                if core < 0:
                    raise ValueError(f"Invalid core index '{item}'")
                if core not in seen:
                    seen.add(core)
                    cores.append(core)

    if len(cores) == 0:
        raise ValueError("No valid CPU cores specified")

    return cores


def set_process_affinity(core_tokens: Iterable[str], pid: int = 0) -> List[int]:

    cores = parse_core_list(core_tokens)
    os.sched_setaffinity(pid, cores)
    return cores

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
