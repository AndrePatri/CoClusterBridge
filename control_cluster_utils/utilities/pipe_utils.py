import os

import yaml

from typing import List
from enum import Enum
from control_cluster_utils.utilities.sysutils import PathsGetter

class NamedPipesHandler:

    OMode = {
    'O_WRONLY': os.O_WRONLY,
    'O_WRONLY_NONBLOCK': os.O_WRONLY | os.O_NONBLOCK,
    'O_RDONLY': os.O_RDONLY,
    'O_RDONLY_NONBLOCK': os.O_RDONLY | os.O_NONBLOCK,
    }

    DSize = {
        'int': 4 # bytes, 

    }

    def __init__(self, 
            name: str = "PipesManager"):

        self.name = name

        self.info = "info"
        self.status = "status"
        self.exception = "exception"
        self.warning = "warning"

        paths = PathsGetter()
        self.pipes_config_path = paths.PIPES_CONFIGPATH

        with open(self.pipes_config_path) as file:
            
            self.yamldata = yaml.load(file, Loader=yaml.FullLoader)

        self.pipes_path = self.yamldata["path"]

        if not os.path.exists(self.pipes_path):
            
            print(f"[{self.name}]" + f"{self.status}" + ": creating pipe directory @" + self.pipes_path)
            os.mkdir(self.pipes_path)

        self.ext = self.yamldata["extension"]

        self.build_pipenames = self.yamldata["build_basenames"]
        self.n_build_pipes = len(self.build_pipenames)
        self.runtime_pipenames = self.yamldata["runtime_basenames"]
        self.n_runtime_pipes = len(self.build_pipenames)

        self.pipenames_all = self.build_pipenames + self.runtime_pipenames

        self.build_pipes_created = False
        self.runtime_pipes_created = False

        self.n_replicae = 0

        self.pipes = {}
        self.buildpipes= {}
        self.pipes_fd = {}
        self.is_open = {}

        self._create_build_pipes()

    def _create_build_pipes(self):
        
        for i in range(0, self.n_build_pipes):
            
            pipe_path = self.pipes_path + "/" + self.build_pipenames[i] + "." + self.ext

            self.pipes[self.build_pipenames[i]] = pipe_path

            self.buildpipes[self.build_pipenames[i]] = pipe_path

            self.pipes_fd[self.build_pipenames[i]] = -1

            self.is_open[self.build_pipenames[i]] = False

            if not os.path.exists(pipe_path):
            
                print(f"[{self.name}]" + f"[{self.status}]" + ": creating pipe @ " + pipe_path)

                os.mkfifo(pipe_path)

        self.build_pipes_created = True

    def _create_runtime_pipes(self, 
            n_replicae: int):

        if not self.runtime_pipes_created: 

            for runtime_pipe in self.runtime_pipenames:
                
                self.pipes[runtime_pipe] = []
                self.pipes_fd[runtime_pipe] = [-1] * n_replicae
                self.is_open[runtime_pipe] = [False] * n_replicae

                for i in range(0, n_replicae):

                    pipe_path = self.pipes_path + "/" + runtime_pipe + f"{i}" + "." + self.ext

                    self.pipes[runtime_pipe].append(pipe_path)

                    if not os.path.exists(pipe_path):
                    
                        print(f"[{self.name}]" + f"[{self.status}]" + ": creating pipe @ " + pipe_path)

                        os.mkfifo(pipe_path)

                self.n_replicae = n_replicae

        else:

            print(f"[{self.name}]" + f"[{self.warning}]" + f"[{self._create_runtime_pipes.__name__}]" + \
                  ": runtime pipes already initialized. This method can only be called once!!")

        self.runtime_pipes_created = True

    def _open(self, 
            pipe: str, 
            mode: OMode, 
            index: int = -1):
        
        if index < 0:

            # build pipe --> we don't use the index

            if not os.path.exists(self.pipes[pipe]):
                
                print(f"[{self.name}]" + f"[{self.warning}]"  + \
                    ": will not open pipe @ " + self.pipes[pipe] + ". It does not exist!!")

                return -1
            
            else:

                print(f"[{self.name}]" + f"[{self.status}]"  +  \
                    f"[{self._open.__name__}]" +": opening pipe @" + self.pipes[pipe])

                self.is_open[pipe] = True

                return os.open(self.pipes[pipe], mode)

        else:

            # runtime pipe --> we use the index

            if not os.path.exists(self.pipes[pipe][index]):
                
                print(f"[{self.name}]" + f"[{self.warning}]" +  f"[{self._open.__name__}]" + \
                    ": will not open pipe @ " + self.pipes[pipe][index] + ". It does not exist!!")

                return -1
            
            else:

                print(f"[{self.name}]" + f"[{self.status}]" +  f"[{self._open.__name__}]" + \
                    ": opening pipe @" + self.pipes[pipe][index])
                
                self.is_open[pipe][index] = True

                return os.open(self.pipes[pipe][index], mode)

    def _close(self, 
            pipe: str, 
            index: int = -1):
        
        if index < 0:

            # build pipe --> we don't use the index

            if not os.path.exists(self.pipes[pipe]):
                
                print(f"[{self.name}]" + f"[{self.warning}]" +  f"[{self._close.__name__}]" + \
                    ": will not close pipe @ " + self.pipes[pipe] + ". It does not exist!!")

                return -1
            
            else:
                
                print(f"[{self.name}]" + f"[{self.status}]" +  f"[{self._close.__name__}]" + \
                    ": closing pipe @" + self.pipes[pipe])

                self.is_open[pipe] = False

                return os.close(self.pipes_fd[pipe])
            
        else:

            # runtime pipe --> we use the index

            if not os.path.exists(self.pipes[pipe][index]):
                
                print(f"[{self.name}]" + f"[{self.warning}]" +  f"[{self._close.__name__}]" +\
                    ": will not close pipe @ " + self.pipes[pipe][index] + ". It does not exist!!")

                return -1
            
            else:

                print(f"[{self.name}]" + f"[{self.status}]" +  f"[{self._close.__name__}]" + \
                    ": closing pipe @" + self.pipes[pipe][index])

                self.is_open[pipe][index] = False

                return os.close(self.pipes_fd[pipe][index])
        
    def open_pipes(self, 
        selector: List[str], 
        mode: OMode, 
        index: int = 0):

        for pipe in selector:

            if pipe in self.buildpipes: # build pipe type
                
                if not self.is_open[pipe]: 

                    self.pipes_fd[pipe] = self._open(pipe, mode)
                
                else:

                    print(f"[{self.name}]" + f"[{self.warning}]" +  f"[{self.open_pipes.__name__}]" + \
                        ": will not open pipe @ " + self.pipes[pipe] + ". It's already opened")

            if (not pipe in self.buildpipes) and (pipe in self.pipes): # runtime pipe

                if not self.is_open[pipe][index]: 

                    self.pipes_fd[pipe][index] = self._open(pipe, mode, index)
                
                else:

                    print(f"[{self.name}]" + f"[{self.warning}]" +  f"[{self.open_pipes.__name__}]" + \
                        ": will not open pipe @ " + self.pipes[pipe][index] + ". It's already opened")

    def close_pipes(self, 
        selector: List[str], 
        index: int = 0):

        for pipe in selector:

            if pipe in self.buildpipes: # build pipe type

                self._close(pipe)

            if (not pipe in self.buildpipes) and (pipe in self.pipes): # runtime pipe

                self._close(pipe, index)

    def create_buildpipes(self):

        self._create_build_pipes()

    def create_runtime_pipes(self, 
                    n_replicae: int):

        self._create_runtime_pipes(n_replicae)



