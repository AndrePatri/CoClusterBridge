import torch
import posix_ipc
import mmap

import yaml

from control_cluster_utils.utilities.sysutils import PathsGetter

class SharedMemConfig:

    def __init__(self):

        paths = PathsGetter()
        self.config_path = paths.SHARED_MEM_CONFIGPATH

        with open(self.config_path) as file:
            
            yamldata = yaml.load(file, Loader=yaml.FullLoader)

        self.basename = yamldata["basename"]

        self.mem_path = "/" + self.basename

class SharedMemSrvr:

    def __init__(self, 
                n_rows: int, 
                n_cols: int,
                name = None, 
                dtype=torch.float32):

        self.dtype = dtype

        self.name = name

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.mem_config = SharedMemConfig()

        if self.name is not None: 
            
            self.mem_config.mem_path = "/" + self.mem_config.basename + \
                self.name
            
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.memory = None

        self.shm = None

        self.create_shared_memory()

        self.create_tensor_view()
                
    def __del__(self):

        self.close_shared_memory()

    def create_shared_memory(self):

        tensor_size = self.n_rows * self.n_cols * self.element_size

        self.shm = posix_ipc.SharedMemory(name = self.mem_config.mem_path, 
                                flags=posix_ipc.O_CREAT, # creates it if not existent
                                size=tensor_size)

        self.memory = mmap.mmap(self.shm.fd, self.shm.size)

        self.shm.close_fd() # we can close the file descriptor (memory remains
        # available for use)
    
    def close_shared_memory(self):

        if self.memory is not None:

            self.memory.close()

            self.memory = None

        if self.shm is not None:

            self.shm.unlink()

            self.shm.close_fd()

    def create_tensor_view(self):

        if self.memory is not None:

            self.tensor_view = torch.frombuffer(self.memory,
                                        dtype=self.dtype, 
                                        count=self.n_rows * self.n_cols).view(self.n_rows, self.n_cols)

class SharedMemClient:

    def __init__(self, 
                n_rows: int, 
                n_cols: int,
                client_index: int, 
                name = 'shared_memory', 
                dtype=torch.float32, 
                verbose = False, 
                wait_amount = 0.05):
        
        self.verbose = verbose

        self.wait_amount = wait_amount

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        self.dtype = dtype

        self.name = name

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.client_index = client_index

        self.mem_config = SharedMemConfig()

        if self.name is not None: 
            
            self.mem_config.mem_path = "/" + self.mem_config.basename + \
                self.name
        
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.memory = None

        self.shm = None

        self.attach_shared_memory()

        self.tensor_view = self.create_tensor_view()

    def __del__(self):

        self.detach_shared_memory()

    def attach_shared_memory(self):

        tensor_size = self.n_rows * self.n_cols * self.element_size
        
        import time

        while True:

            try:

                self.shm = posix_ipc.SharedMemory(name = self.mem_config.mem_path, 
                                size=tensor_size)

                break  # exit loop if attached successfully

            except posix_ipc.ExistentialError:
                
                if self.verbose: 

                    status = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                                    f"[{self.status}]" + ":" + f"waiting for memory at {self.mem_config.mem_path} to be allocated by the server..."
                    
                    print(status)

                time.sleep(self.wait_amount)

                continue

        self.memory = mmap.mmap(self.shm.fd, self.shm.size)

        self.shm.close_fd() 

    def create_tensor_view(self, 
                index: int = None, 
                length: int = None):

        if self.memory is not None:
            
            if (index is None) or (length is None):

                offset = self.client_index * self.n_cols * self.element_size
                
                return torch.frombuffer(self.memory,
                                dtype=self.dtype, 
                                count=self.n_cols, 
                                offset=offset).view(1, self.n_cols)
            else:
                
                if index >= self.n_cols:

                    exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                                    f"[{self.exception}]" + f"[{self.create_tensor_view.__name__}]" + \
                                    ":" + f"the provided index {index} exceeds {self.n_cols - 1}"
                
                    raise ValueError(exception)
                
                if length > (self.n_cols - index):

                    exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                                    f"[{self.exception}]" + f"[{self.create_tensor_view.__name__}]" + \
                                    ":" + f"the provided length {length} exceeds {(self.n_cols - index)}"
                
                    raise ValueError(exception)
                
                offset = self.client_index * self.n_cols * self.element_size + \
                    index * self.element_size 
                
                return torch.frombuffer(self.memory,
                                dtype=self.dtype, 
                                count=length, 
                                offset=offset).view(1, length)

    def detach_shared_memory(self):

        if self.memory is not None:

            self.memory.close()

            self.memory = None

        if self.shm is not None:

            self.shm.close_fd()
