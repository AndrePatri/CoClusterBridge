import torch

import numpy as np

import posix_ipc
import mmap
import struct

import yaml

from control_cluster_utils.utilities.sysutils import PathsGetter
from control_cluster_utils.utilities.defs import shared_clients_count_name
from control_cluster_utils.utilities.defs import shared_sem_srvr_name
from control_cluster_utils.utilities.defs import shared_sem_clients_count_name

from typing import List

import time

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
                dtype=torch.float32, 
                backend="torch"):

        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        self.backend = backend

        self.dtype = dtype

        self.name = name

        self.n_rows = n_rows
        self.n_cols = n_cols

        self._terminate = False 
        self._started = False

        self.mem_config = SharedMemConfig()

        if self.name is not None: 
            
            self.mem_config.mem_path = "/" + self.mem_config.basename + \
                self.name
            
            self.mem_path_clients_counter = "/" + self.mem_config.basename + \
                self.name + "_" + shared_clients_count_name()
            
            self.mem_path_server_sem = "/" + self.mem_config.basename + \
                self.name + "_" + shared_sem_srvr_name()
            
            self.mem_path_clients_n_sem = "/" + self.mem_config.basename + \
                self.name + "_" + shared_sem_clients_count_name()
            
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        # shared mem fd
        self.shm = None
        self.shm_clients_counter = None
        self.sem_server = None
        self.sem_client_n = None

        # mem maps
        self.memory = None
        self.memory_clients_counter = None

        # views
        self.bool_bytearray_view = None
        self.tensor_view = None
        self.n_clients = None

        self.torch_to_np_dtype = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: bool,
            }

        self._create_shared_memory()

    def start(self):
        
        if not self.is_server_unique():
                        
            raise Exception(self.mult_srvrs_error())
        
        self._create_tensor_view()
        
        self.bool_bytearray_view = None

        if self.backend == "torch": 

            if self.dtype == torch.bool and \
                (self.n_rows == 1 or self.n_cols == 1):
                
                self._create_bytearray_view() # more efficient array view for simple 1D boolean 
                # arrays

        if self.backend == "numpy":

            if self.dtype == np.bool and \
                (self.n_rows == 1 or self.n_cols == 1):

                self._create_bytearray_view()
        
        self._started = True
                
    def __del__(self):

        self.terminate()

    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self._close_shared_memory()

    def print_created_mem(self, 
                        path: str):
        
        message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._create_shared_memory.__name__}]: " + \
                    f"created shared memory of datatype {self.dtype}" + \
                    f", size {self.n_rows} x {self.n_cols} @ {path}"
        
        print(message)

    def mult_srvrs_error(self):

        error = f"\n[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._create_shared_memory.__name__}]: " + \
                        f"a server @ {self.mem_config.mem_path} already exists. Only one server can be created at a time!\n"

        return error
        
    def _create_semaphores(self):

        self.sem_client_n = posix_ipc.Semaphore(self.mem_path_clients_n_sem, 
                            flags=posix_ipc.O_CREAT, 
                            initial_value=1)
        
        message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._create_semaphores.__name__}]: " + \
                    f"created/opened sempaphore @ {self.mem_path_clients_n_sem}"
        
        print(message)

        self.sem_server = posix_ipc.Semaphore(self.mem_path_server_sem, 
                            flags=posix_ipc.O_CREAT, 
                            initial_value=1)
        
        message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._create_semaphores.__name__}]: " + \
                    f"created/opened sempaphore @ {self.mem_path_server_sem}"
        
        print(message)

    def is_server_unique(self):
        
        try:

            self.sem_server.acquire(timeout=0.1)

            return True
        
        except posix_ipc.BusyError: # failed to acquire --> another server
            
            return False
        
    def _create_clients_counter(self):

        self.shm_clients_counter = posix_ipc.SharedMemory(name = self.mem_path_clients_counter, 
                                flags=posix_ipc.O_CREAT,
                                size=8) # each client will increment this counter

        self.memory_clients_counter = mmap.mmap(self.shm_clients_counter.fd, self.shm_clients_counter.size)

        self.shm_clients_counter.close_fd()

        self.n_clients = memoryview(self.memory_clients_counter)

        message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._create_clients_counter.__name__}]: " + \
                    f"created/opened clients counter @ {self.mem_path_clients_counter}"
        
        print(message)

    def get_clients_count(self):
        
        if self.n_clients is not None:
            
            while not self._terminate:

                try:

                    # self.sem_client_n.acquire(timeout=0.0) 

                    count = struct.unpack('q', self.n_clients[:8])[0]
                    
                    # self.sem_client_n.release()

                    return count 
                
                except posix_ipc.BusyError:

                    continue
        
    def _create_data_memory(self, 
                    tensor_size: int):

        self.shm = posix_ipc.SharedMemory(name = self.mem_config.mem_path, 
                                flags=posix_ipc.O_CREAT, # creates it if not existent
                                size=tensor_size) 
        self.print_created_mem(self.mem_config.mem_path)
        
        self.memory = mmap.mmap(self.shm.fd, self.shm.size)

        self.shm.close_fd() # we can close the file descriptor (memory remains
        # available for use)

    def _create_shared_memory(self):

        tensor_size = self.n_rows * self.n_cols * self.element_size

        # semaphore
        self._create_semaphores()
    
        # clients counter
        self._create_clients_counter()

        # data memory
        self._create_data_memory(tensor_size)
    
    def _detach_vars(self):

        if self.bool_bytearray_view is not None:

            self.bool_bytearray_view = None

        if self.tensor_view is not None:

            self.tensor_view = None
        
        if self.n_clients is not None:

            self.n_clients = None

    def _close_mmaps(self):
        
        if self.memory is not None:

            self.memory.close()

            self.memory = None

            message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._close_mmaps.__name__}]: " + \
                        f"closed shared memory of datatype {self.dtype}" + \
                        f", size {self.n_rows} x {self.n_cols} @ {self.mem_config.mem_path}"
        
            print(message)

        if self.memory_clients_counter is not None:
            
            self.memory_clients_counter.close()

            self.memory_clients_counter = None

            message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self._close_mmaps.__name__}]: " + \
                        f"closed clients counter @ {self.mem_path_clients_counter}"
        
            print(message)

    def _close_sems(self):

        if self.sem_server is not None:
            
            try:
                
                self.sem_server.release()

                self.sem_server.close()

                self.sem_server.unlink()

            except posix_ipc.ExistentialError:
                
                message = f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + f"[{self._unlink.__name__}]: " + \
                        f"could not close semaphor @ {self.mem_path_server_sem}. Probably something already did."

                print(message)

                pass
        
        if self.sem_client_n is not None:

            try:
                
                self.sem_client_n.release()

                self.sem_client_n.close()

                self.sem_client_n.unlink() # should we unlink it?

            except posix_ipc.ExistentialError:
                
                message = f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + f"[{self._unlink.__name__}]: " + \
                        f"could not close semaphor @ {self.mem_path_clients_n_sem}. Probably something already did."

                print(message)

                pass

    def _unlink(self):

        if self.shm is not None:
            
            try:
                
                self.shm.unlink() # this removes the shared memory segment from the system

                self.shm.close_fd()

                self.shm = None

            except posix_ipc.ExistentialError:
                
                message = f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + f"[{self._unlink.__name__}]: " + \
                        f"could not unlink shared memory of datatype {self.dtype}" + \
                        f", size {self.n_rows} x {self.n_cols} @ {self.mem_config.mem_path}. Probably something already did."

                print(message)

                pass

        if self.shm_clients_counter is not None:

            try:

                # self.shm_clients_counter.unlink() # should we unlink it?

                self.shm_clients_counter.close_fd()

                self.shm_clients_counter = None

            except posix_ipc.ExistentialError:
                
                message = f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + f"[{self._unlink.__name__}]: " + \
                        f"could not unlink clients counter @ {self.mem_path_clients_counter}. Probably something already did."

                print(message)

                pass

    def _close_shared_memory(self):
        
        self._close_sems()

        self._detach_vars()

        self._close_mmaps()

        self._unlink()

    def _create_tensor_view(self):

        if self.memory is not None:
            
            if self.backend == "torch":

                self.tensor_view = torch.frombuffer(self.memory,
                                            dtype=self.dtype, 
                                            count=self.n_rows * self.n_cols).view(self.n_rows, self.n_cols)

            if self.backend == "numpy":

                self.tensor_view = np.frombuffer(self.memory,
                                            dtype=self.torch_to_np_dtype[self.dtype], 
                                            count=self.n_rows * self.n_cols).reshape(self.n_rows, self.n_cols).view()
    
    def _create_bytearray_view(self):

        self.bool_bytearray_view = memoryview(self.memory)

        self.bytearray_reset_false = bytearray(len(self.bool_bytearray_view))

        self.bytearray_reset_true = bytearray([1] * len(self.bool_bytearray_view))

    def all(self):

        if self.bool_bytearray_view is not None:

            return all(value == 1 for value in self.bool_bytearray_view)
        
        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
    def none(self):

        if self.bool_bytearray_view is not None:

            return all(value == 0 for value in self.bool_bytearray_view)
        
        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.none.__name__}]" + \
                            ":" + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)

    def reset_bool(self, 
                to_true: bool = False):

        # sets all to either True or False
        if self.bool_bytearray_view is not None:
            
            if to_true:

                self.bool_bytearray_view[:] = self.bytearray_reset_true

            else:

                self.bool_bytearray_view[:] = self.bytearray_reset_false

        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
    def set_bool(self, 
                vals: List[bool]):

        # sets all to either True or False
        if self.bool_bytearray_view is not None:
            
            if len(vals) != len(self.bool_bytearray_view):

                exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f" provided boolean list of length {len(vals)} does" + \
                            f" not match the required lentgh of {len(self.bool_bytearray_view)}"

                raise Exception(exception)
        
            self.bool_bytearray_view[:] = bytearray(vals)

        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
class SharedMemClient:

    def __init__(self, 
                n_rows: int, 
                n_cols: int,
                name: str, 
                client_index: int = None, 
                dtype=torch.float32, 
                backend="torch",
                verbose = False, 
                wait_amount = 0.05):
        
        self.backend = backend

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
            
            self.mem_path_clients_counter = "/" + self.mem_config.basename + \
                self.name + "_" + shared_clients_count_name()
            
            self.mem_path_clients_semaphore = "/" + self.mem_config.basename + \
                self.name + "_" + shared_sem_clients_count_name()
        
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.memory = None
        self.memory_clients_counter = None
        self.sem_client_n = None

        self.shm = None
        self.shm_clients_counter = None
        
        self.bool_bytearray_view = None
        self.tensor_view = None
        self.n_clients = None

        self._terminate = False
        self._attached = False

    def __del__(self):
        
        self.terminate()

    def terminate(self):

        if not self._terminate:
            
            self._decrement_client_count()

            self._terminate = True

            self._detach_shared_memory()

    def attach(self):
        
        self._attach_shared_memory()
        
        if self._attached:

            if self.backend == "torch": 

                if self.dtype == torch.bool and \
                    (self.n_rows == 1 or self.n_cols == 1):
                    
                    self.is_bool_mode = True

                    if self.n_rows == 1:

                        self.client_index = 0 # this way we allow multiple "subscribers"
                        # to a global boolean var
                    
                    self._create_bytearray_view() # more efficient array view for simple 1D boolean 
                    # arrays

            if self.backend == "numpy":

                if self.dtype == np.bool and \
                    (self.n_rows == 1 or self.n_cols == 1):

                    self.is_bool_mode = True
                    
                    if self.n_rows == 1:

                        self.client_index = 0 # this way we allow multiple "subscribers"
                        # to a global boolean var
                        
                    self._create_bytearray_view()
            
            if self.client_index is not None:
                
                self.tensor_view = self.create_partial_tensor_view() # creates a view of a whole row

            else:
                
                self.tensor_view = self._create_tensor_view()

            return True
        
        else:

            return False

    def _print_wait(self, 
                path: str):
        
        index = "" if self.client_index is None else self.client_index
        status = "[" + self.__class__.__name__ + str(self.name) + str(index) + "]"  + \
                        f"[{self.status}]" + ": " + \
                            f"waiting for memory at {path} to be allocated by the server..."
                    
        print(status)
    
    def _print_attached(self, 
                    path: str):

        index = "" if self.client_index is None else self.client_index
        message = "[" + self.__class__.__name__ + str(self.name) + str(index) + "]"  + \
                                f"[{self.status}]" + ": " + f"attaced to memory @ {path}."
        
        print(message)

    def _print_detached(self, 
                    path: str):

        index = "" if self.client_index is None else self.client_index
        message = "[" + self.__class__.__name__ + str(self.name) + str(index) + "]"  + \
                                f"[{self.status}]" + ": " + f"detached from memory @ {path}."
        
        print(message)

    def _handle_posix_error(self, 
                        path: str):

        if self.verbose: 
                
            self._print_wait(path)
            
        time.sleep(self.wait_amount)

    def _attach_clients_counter(self):

        while not self._terminate:
            
            try:

                self.shm_clients_counter = posix_ipc.SharedMemory(name = self.mem_path_clients_counter, 
                            size=8) # each client will increment this counter

                self._print_attached(self.mem_path_clients_counter)

                self.memory_clients_counter = mmap.mmap(self.shm_clients_counter.fd, self.shm_clients_counter.size)
                self.shm_clients_counter.close_fd()
                
                time.sleep(self.wait_amount)

                self.n_clients = memoryview(self.memory_clients_counter)

                return True

            except posix_ipc.ExistentialError: 
            
                self._handle_posix_error(self.mem_path_clients_counter)

                continue

            except KeyboardInterrupt:

                self.terminate()
            
    def _attach_semaphore(self):
        
        while not self._terminate:

            try:

                self.sem_client_n = posix_ipc.Semaphore(self.mem_path_clients_semaphore, 
                    initial_value=1)

                self._print_attached(self.mem_path_clients_semaphore)

                time.sleep(self.wait_amount)

                return True

            except posix_ipc.ExistentialError:
            
                self._handle_posix_error(self.mem_path_clients_semaphore)
                
                continue

            except KeyboardInterrupt:

                self.terminate()
            
    def _attach_shared_data(self, 
                    tensor_size: int):

        while not self._terminate:

            try:
                
                self.shm = posix_ipc.SharedMemory(name = self.mem_config.mem_path, 
                            size=tensor_size)

                self.memory = mmap.mmap(self.shm.fd, self.shm.size)

                self.shm.close_fd() 
                
                self._print_attached(self.mem_config.mem_path)
                                    
                time.sleep(self.wait_amount)

                return True

            except posix_ipc.ExistentialError: 
            
                self._handle_posix_error(self.mem_config.mem_path)
                    
                continue

            except KeyboardInterrupt:

                self.terminate()
                
    def _attach_shared_memory(self):

        tensor_size = self.n_rows * self.n_cols * self.element_size

        success_sem = self._attach_semaphore()

        success_counter = self._attach_clients_counter()
                
        success_data = self._attach_shared_data(tensor_size)

        if success_sem and success_counter and success_data:

            self._attached = True
        
        else:

            self._attached = False

        self._increment_client_count() # only works if successfully attached
        
    def _create_tensor_view(self):

        if self.memory is not None:
            
            if self.backend == "torch":

                return torch.frombuffer(self.memory,
                                dtype=self.dtype, 
                                count=self.n_rows * self.n_cols).view(self.n_rows, self.n_cols)

            if self.backend == "numpy":

                return np.frombuffer(self.memory,
                            dtype=self.torch_to_np_dtype[self.dtype], 
                            count=self.n_rows * self.n_cols).reshape(self.n_rows, self.n_cols).view()
    
        else:

            return None
        
    def create_partial_tensor_view(self, 
                index: int = None, 
                length: int = None):

        if self.memory is not None:
            
            if self.client_index is not None:

                if self.backend == "torch":

                    if (index is None) or (length is None):
                    
                        offset = self.client_index * self.n_cols * self.element_size
                        
                        return torch.frombuffer(self.memory,
                                        dtype=self.dtype, 
                                        count=self.n_cols, 
                                        offset=offset).view(1, self.n_cols)
                        
                    else:
                        
                        if index >= self.n_cols:

                            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                                            f"[{self.exception}]" + f"[{self.create_partial_tensor_view.__name__}]" + \
                                            ": " + f"the provided index {index} exceeds {self.n_cols - 1}"
                        
                            raise ValueError(exception)
                        
                        if length > (self.n_cols - index):

                            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                                            f"[{self.exception}]" + f"[{self.create_partial_tensor_view.__name__}]" + \
                                            ": " + f"the provided length {length} exceeds {(self.n_cols - index)}"
                        
                            raise ValueError(exception)
                        
                        offset = self.client_index * self.n_cols * self.element_size + \
                            index * self.element_size 
                        
                        return torch.frombuffer(self.memory,
                                        dtype=self.dtype, 
                                        count=length, 
                                        offset=offset).view(1, length)
                
                if self.backend == "numpy":

                    if (index is None) or (length is None):

                        offset = self.client_index * self.n_cols * self.element_size
                        
                        return np.frombuffer(self.memory,
                                        dtype=self.torch_to_np_dtype[self.dtype], 
                                        count=self.n_cols, 
                                        offset=offset).reshape(1, self.n_cols).view()
                    else:
                        
                        if index >= self.n_cols:

                            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                                            f"[{self.exception}]" + f"[{self.create_partial_tensor_view.__name__}]" + \
                                            ": " + f"the provided index {index} exceeds {self.n_cols - 1}"
                        
                            raise ValueError(exception)
                        
                        if length > (self.n_cols - index):

                            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                                            f"[{self.exception}]" + f"[{self.create_partial_tensor_view.__name__}]" + \
                                            ": " + f"the provided length {length} exceeds {(self.n_cols - index)}"
                        
                            raise ValueError(exception)
                        
                        offset = self.client_index * self.n_cols * self.element_size + \
                            index * self.element_size 
                        
                        return np.frombuffer(self.memory,
                                        dtype=self.dtype, 
                                        count=length, 
                                        offset=offset).reshape(1, length).view()
            
            else:

                exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                                f"[{self.exception}]" + f"[{self.create_partial_tensor_view.__name__}]" + \
                                ": " + f"can only call create_partial_tensor_view is " + \
                                "a client_index was provided upon initialization."
                    
                raise Exception(exception)
            
    def _create_bytearray_view(self):

        self.bool_bytearray_view = memoryview(self.memory)
        
    def set_bool(self, 
                val: bool = False):

        if self.bool_bytearray_view is not None:
            
            if self.client_index >= 0 and \
                self.client_index < len(self.bool_bytearray_view) and \
                len(self.bool_bytearray_view) > 1:

                self.bool_bytearray_view[self.client_index] = val

            else:

                exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"bool val will not be assigned for dim incompatibility."

                print(exception)
        
        else:

            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
    
    def read_bool(self):
    
        if self.bool_bytearray_view is not None:
            
            if self.client_index >= 0 and \
                self.client_index < len(self.bool_bytearray_view) and \
                len(self.bool_bytearray_view) > 1:

                return self.bool_bytearray_view[self.client_index]

            if self.client_index < 0 or \
                self.client_index >= len(self.bool_bytearray_view):

                exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"bool val will not be read for dim incompatibility."

                raise Exception(exception)

            if len(self.bool_bytearray_view) == 1:

                return self.bool_bytearray_view[0]
            
        else:

            exception = f"[{self.__class__.__name__}{self.client_index}]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ": " + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
    
    def _increment_client_count(self):
        
        if self._attached:
                        
            while not self._terminate:
                
                try:
                    
                    self.sem_client_n.acquire(timeout=0) # we ensure exclusive access when writing 

                    current_value = struct.unpack('q', self.n_clients[:8])[0]  # we read the whole int64 (8 bytes)

                    self.n_clients[:8] = struct.pack('q', current_value + 1) 

                    self.sem_client_n.release()

                    break

                except posix_ipc.BusyError: 
                    
                    time.sleep(self.wait_amount)

                    if self.verbose:
                        
                        index = "" if self.client_index is None else self.client_index

                        message = "[" + self.__class__.__name__ + str(self.name) + str(index) + "]"  + \
                                f"[{self.status}]" +  f"[{self._increment_client_count.__name__}]" + ": " + \
                                f" failed to acquire semaphore @ {self.mem_path_clients_semaphore}." + \
                                " Retrying ..."

                        print(message)

                        continue

    def _decrement_client_count(self):
        
        if self._attached:

            while not self._terminate:
                
                try:

                    self.sem_client_n.acquire(timeout=0) # we ensure exclusive access when writing

                    current_value = struct.unpack('q', self.n_clients[:8])[0]  # we read the whole int64 (8 bytes)

                    self.n_clients[:8] = struct.pack('q', current_value - 1) 

                    self.sem_client_n.release()

                    break
                
                except posix_ipc.BusyError: 

                    time.sleep(self.wait_amount)

                    if self.verbose:
                        
                        index = "" if self.client_index is None else self.client_index

                        message = "[" + self.__class__.__name__ + str(self.name) + str(index) + "]"  + \
                                f"[{self.status}]" +  f"[{self._decrement_client_count.__name__}]" + ": " + \
                                f" failed to acquire semaphore @ {self.mem_path_clients_semaphore}." + \
                                " Retrying ..."

                        print(message)

                        continue

    def _detach_shared_memory(self):
        
        if self.bool_bytearray_view is not None:

            self.bool_bytearray_view = None

        if self.tensor_view is not None:

            self.tensor_view = None

        if self.n_clients is not None:

            self.n_clients = None

        if self.memory is not None:
            
            self.memory.close() # memory remains available to other clients

            self.memory = None

            self._print_detached(self.mem_config.mem_path)
            
        if self.shm is not None:

            self.shm.close_fd()

        if self.memory_clients_counter is not None:
                        
            self.n_clients = None # we destroy memory view

            self.memory_clients_counter.close()

            self.memory_clients_counter = None

            self._print_detached(self.mem_path_clients_counter)
                
        if self.shm_clients_counter is not None:

            self.shm_clients_counter.close_fd()

        if self.sem_client_n is not None:
            
            self.sem_client_n = None

            self._print_detached(self.mem_path_clients_semaphore)
            
class SharedStringArray:

    # not specifically designed to be low-latency
    # (should be used sporadically or for initialization purposes)

    def __init__(self, 
            length: int, 
            name: str, 
            is_server: bool,
            init: List[str] = None, 
            verbose: bool = False, 
            wait_amount: float = 0.01):
        
        self.verbose = verbose

        self.length = length
        
        self.is_server = is_server

        self._terminate = False

        self.dtype = torch.int64
        self.max_string_length = 64 # num of characters for each string
        self.chunk_size = torch.tensor([0],dtype=self.dtype).element_size()

        import math
        self.n_rows = math.ceil(self.max_string_length / 
                            torch.tensor([0],dtype=self.dtype).element_size())
        self.basename = f"{self.__class__.__name__}"

        self.name = self.basename + name
        
        self.status = "status"
        self.info = "info"
        self.exception = "exception"
        self.warning = "warning"

        if self.is_server:

            self.mem_manager = SharedMemSrvr(self.n_rows, 
                                        self.length, 
                                        self.name, 
                                        dtype=self.dtype)
        else:

            self.mem_manager = SharedMemClient(n_rows=self.n_rows, 
                                        n_cols=self.length, 
                                        name=self.name, 
                                        dtype=self.dtype, 
                                        verbose=verbose, 
                                        wait_amount=wait_amount)
    
    def start(self, 
            init: List[str] = None):

        if self.is_server:

            self.mem_manager.start()

        else:

            self.mem_manager.attach()

        if init is not None:

            self.write(init)

    def __del__(self):

        self.terminate()
    
    def terminate(self):
        
        if not self._terminate:

            self._terminate = True
            
            self.mem_manager.terminate()

    def write(self, 
        lst: List[str]):

        if self.is_server:

            lst = self._check_list(lst)

            self._encode(lst)

        else:

            message = f"[{self.__class__.__name__}]"  + f"[{self.warning}]" + f"[{self.write.__name__}]: " + \
                        f"can only call the write() method on server!!"
        
            print(message)

    def read(self):

        return self._decode()

    def _split_into_chunks(self, 
                input_string: str, 
                chunk_size: int, 
                num_chunks: int):

        chunks = [input_string[i:i+chunk_size] for i in range(0, len(input_string), chunk_size)]
        if len(chunks) < num_chunks:
            chunks.extend([''] * (num_chunks - len(chunks)))
        return chunks

    def _encode(self, 
            lst: List[str]):

        for i in range(0, self.length):
            
            chunks = self._split_into_chunks(lst[i], 
                        self.chunk_size, 
                        self.n_rows)
            
            for j in range(0, len(chunks)):
                    
                int_encoding = int.from_bytes(chunks[j].encode('utf-8'), 
                                byteorder='big') 
                
                self.mem_manager.tensor_view[j, i] = torch.tensor(int_encoding, 
                                                                dtype=self.dtype) 

    def _decode(self):
        
        decoded_list = []

        for i in range(0, self.length):
            
            chunks = []

            for j in range(0, self.n_rows):

                int_encoding = self.mem_manager.tensor_view[j, i].item()

                bytes_decoded = int_encoding.to_bytes((int_encoding.bit_length() + 7) // 8, byteorder='big')

                chunks.append(bytes_decoded.decode('utf-8'))
            
            separator = ''  # Separator between each element
            decoded_list.append(separator.join(chunks))

        return decoded_list
    
    def _flatten_recursive(self, 
                    lst: List[str]):
        
        flat_list = []

        for item in lst:

            if isinstance(item, list):

                flat_list.extend(self._flatten_recursive(item))

            else:

                flat_list.append(item)

        return flat_list

    def _check_list(self, lst: List[str]):

        if self._is_more_than_one_dimensional(lst):
            
            lst = self._flatten_recursive(lst)
    
        # we check its dimensions
        self._is_coherent(lst) # will raise exceptions
        
        return lst
    
    def _is_coherent(self,
                    lst: List[str]):
        
        if len(lst) != self.length:

            exception = "[" + self.__class__.__name__ +  "]"  + \
                f"[{self.exception}]" + ": " + f" length mismatch in provided list. It's {len(lst)} but should be {self.length}."
            
            raise Exception(exception)
        
        return True
        
    def _is_list_uniform(self, 
                        lst: List[str]):

        # Get the length of the first internal list
        reference_length = len(lst[0])

        for sublist in lst:
            if len(sublist) != reference_length:

                return False

        return True

    def _get_list_depth(self, 
                    lst: List[str]):

        if not isinstance(lst, list):

            return 0

        max_depth = 0
        for item in lst:

            depth = self._get_list_depth(item)

            max_depth = max(max_depth, depth)

        return max_depth + 1

    def _is_more_than_one_dimensional(self, 
                                lst: List[str]):

        return self._get_list_depth(lst) > 1

    def _is_two_dimensional(self, 
                        lst: List[str]):

        return self._get_list_depth(lst) == 2

    def _is_one_dimensional(self, 
                        lst: List[str]):

        return self._get_list_depth(lst) == 1