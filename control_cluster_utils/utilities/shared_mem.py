import torch

import numpy as np

import posix_ipc
import mmap

import yaml

from control_cluster_utils.utilities.sysutils import PathsGetter

from typing import List

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

        self.mem_config = SharedMemConfig()

        if self.name is not None: 
            
            self.mem_config.mem_path = "/" + self.mem_config.basename + \
                self.name
            
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.memory = None

        self.shm = None

        self.bool_bytearray_view = None

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

        self.create_shared_memory()

        self.create_tensor_view()
        
        self.bool_bytearray_view = None

        if self.backend == "torch": 

            if self.dtype == torch.bool and \
                (self.n_rows == 1 or self.n_cols == 1):
                
                self.create_bytearray_view() # more efficient array view for simple 1D boolean 
                # arrays

        if self.backend == "numpy":

            if self.dtype == np.bool and \
                (self.n_rows == 1 or self.n_cols == 1):

                self.create_bytearray_view()
                
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

        message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self.create_shared_memory.__name__}]: " + \
                        f": created shared memory of datatype {self.dtype}" + \
                        f", size {self.n_rows} x {self.n_cols} @ {self.mem_config.mem_path}"
        
        print(message)

    def close_shared_memory(self):
        
        if self.bool_bytearray_view is not None:

            self.bool_bytearray_view = None

        if self.memory is not None:

            self.memory.close()

            self.memory = None

            message = f"[{self.__class__.__name__}]"  + f"[{self.status}]" + f"[{self.create_shared_memory.__name__}]: " + \
                        f": closed shared memory of datatype {self.dtype}" + \
                        f", size {self.n_rows} x {self.n_cols} @ {self.mem_config.mem_path}"
        
            print(message)

        if self.shm is not None:
            
            try:
                
                self.shm.unlink()

                self.shm.close_fd()

            except posix_ipc.ExistentialError:

                pass

    def create_tensor_view(self):

        if self.memory is not None:
            
            if self.backend == "torch":

                self.tensor_view = torch.frombuffer(self.memory,
                                            dtype=self.dtype, 
                                            count=self.n_rows * self.n_cols).view(self.n_rows, self.n_cols)

            if self.backend == "numpy":

                self.tensor_view = np.frombuffer(self.memory,
                                            dtype=self.torch_to_np_dtype[self.dtype], 
                                            count=self.n_rows * self.n_cols).reshape(self.n_rows, self.n_cols).view()
    
    def create_bytearray_view(self):

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
                            ":" + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
    def set_bool(self, 
                vals: List[bool]):

        # sets all to either True or False
        if self.bool_bytearray_view is not None:
            
            if len(vals) != len(self.bool_bytearray_view):

                exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f" provided boolean list of length {len(vals)} does" + \
                            f" not match the required lentgh of {len(self.bool_bytearray_view)}"

                raise Exception(exception)
        
            self.bool_bytearray_view[:] = bytearray(vals)

        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
class SharedMemClient:

    def __init__(self, 
                n_rows: int, 
                n_cols: int,
                client_index: int, 
                name = 'shared_memory', 
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
        
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.memory = None

        self.shm = None

        self.bool_bytearray_view = None

        self.attach_shared_memory()

        if self.backend == "torch": 

            if self.dtype == torch.bool and \
                (self.n_rows == 1 or self.n_cols == 1):
                
                self.is_bool_mode = True

                if self.n_rows == 1:

                    self.client_index = 0 # this way we allow multiple "subscribers"
                    # to a global boolean var
                
                self.create_bytearray_view() # more efficient array view for simple 1D boolean 
                # arrays

        if self.backend == "numpy":

            if self.dtype == np.bool and \
                (self.n_rows == 1 or self.n_cols == 1):

                self.is_bool_mode = True
                
                if self.n_rows == 1:

                    self.client_index = 0 # this way we allow multiple "subscribers"
                    # to a global boolean var
                    
                self.create_bytearray_view()

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

                time.sleep(self.wait_amount)

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
            
            if self.backend == "torch":

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
                
            if self.backend == "numpy":

                if (index is None) or (length is None):

                    offset = self.client_index * self.n_cols * self.element_size
                    
                    return np.frombuffer(self.memory,
                                    dtype=self.torch_to_np_dtype[self.dtype], 
                                    count=self.n_cols, 
                                    offset=offset).reshape(1, self.n_cols).view()
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
                    
                    return np.frombuffer(self.memory,
                                    dtype=self.dtype, 
                                    count=length, 
                                    offset=offset).reshape(1, length).view()
                
    def create_bytearray_view(self):

        self.bool_bytearray_view = memoryview(self.memory)
        
    def set_bool(self, 
                val: bool = False):

        if self.bool_bytearray_view is not None:
            
            if self.client_index >= 0 and \
                self.client_index < len(self.bool_bytearray_view) and \
                len(self.bool_bytearray_view) > 1:

                self.bool_bytearray_view[self.client_index] = val

            else:

                exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"bool val will not be assigned for dim incompatibility."

                print(exception)
        
        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"no bytearray view available." + \
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

                exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"bool val will not be read for dim incompatibility."

                raise Exception(exception)

            if len(self.bool_bytearray_view) == 1:

                return self.bool_bytearray_view[0]
            
        else:

            exception = "[" + self.__class__.__name__ + str(self.client_index) + "]"  + \
                            f"[{self.exception}]" + f"[{self.all.__name__}]" + \
                            ":" + f"no bytearray view available." + \
                            "Did you initialize the class with boolean dtype and at least one dimension = 1?"

            raise Exception(exception)
        
    def detach_shared_memory(self):
        
        if self.bool_bytearray_view is not None:

            self.bool_bytearray_view = None

        if self.memory is not None:

            self.memory.close()

            self.memory = None

        if self.shm is not None:

            self.shm.close_fd()

class SharedStringArray:

    # not specifically designed to be low-latency
    # (should be used sporadically or for initialization purposes)

    def __init__(self, 
            length: int, 
            name: str, 
            init: List[str] = None):
        
        self.length = length
        
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
    
        self.mem_manager = SharedMemSrvr(self.n_rows, 
                                    self.length, 
                                    self.name, 
                                    dtype=self.dtype)

        if init is not None:

            self.write(init)

    def split_into_chunks(self, 
                input_string: str, 
                chunk_size: int, 
                num_chunks: int):

        chunks = [input_string[i:i+chunk_size] for i in range(0, len(input_string), chunk_size)]
        if len(chunks) < num_chunks:
            chunks.extend([''] * (num_chunks - len(chunks)))
        return chunks

    def encode(self, 
            lst: List[str]):

        for i in range(0, self.length):
            
            chunks = self.split_into_chunks(lst[i], 
                        self.chunk_size, 
                        self.n_rows)
            
            for j in range(0, len(chunks)):
                    
                int_encoding = int.from_bytes(chunks[j].encode('utf-8'), 
                                byteorder='big') 
                
                self.mem_manager.tensor_view[j, i] = torch.tensor(int_encoding, 
                                                                dtype=self.dtype) 

    def decode(self):
        
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
    
    def write(self, 
            lst: List[str]):

        lst = self.check_list(lst)

        self.encode(lst)

    def read(self):

        return self.decode()

    def flatten_recursive(self, 
                    lst: List[str]):
        
        flat_list = []

        for item in lst:

            if isinstance(item, list):

                flat_list.extend(self.flatten_recursive(item))

            else:

                flat_list.append(item)

        return flat_list

    def check_list(self, lst: List[str]):

        if self.is_more_than_one_dimensional(lst):
            
            lst = self.flatten_recursive(lst)
    
        # we check its dimensions
        self.is_coherent(lst) # will raise exceptions
        
        return lst
    
    def is_coherent(self,
                    lst: List[str]):
        
        if len(lst) != self.length:

            exception = "[" + self.__class__.__name__ +  "]"  + \
                f"[{self.exception}]" + ":" + f" length mismatch in provided list. It's {len(lst)} but should be {self.length}."
            
            raise Exception(exception)
        
        return True
        
    def is_list_uniform(self, 
                        lst: List[str]):

        # Get the length of the first internal list
        reference_length = len(lst[0])

        for sublist in lst:
            if len(sublist) != reference_length:

                return False

        return True

    def get_list_depth(self, 
                    lst: List[str]):

        if not isinstance(lst, list):

            return 0

        max_depth = 0
        for item in lst:

            depth = self.get_list_depth(item)

            max_depth = max(max_depth, depth)

        return max_depth + 1

    def is_more_than_one_dimensional(self, 
                                lst: List[str]):

        return self.get_list_depth(lst) > 1

    def is_two_dimensional(self, 
                        lst: List[str]):

        return self.get_list_depth(lst) == 2

    def is_one_dimensional(self, 
                        lst: List[str]):

        return self.get_list_depth(lst) == 1