from abc import ABC, abstractmethod

def flatten_shared_mem(shared_mem):

    if shared_mem is None:
        return []

    if isinstance(shared_mem, (list, tuple)):
        flattened = []
        for item in shared_mem:
            flattened.extend(flatten_shared_mem(item))
        return flattened

    return [shared_mem]

def infer_shm_type(shared_mem):

    type_fqn = f"{type(shared_mem).__module__}.{type(shared_mem).__name__}"
    if type_fqn in (
        "EigenIPC.PyEigenIPC.StringTensorClient",
        "EigenIPC.PyEigenIPC.StringTensorServer",
        "PyEigenIPC.StringTensorClient",
        "PyEigenIPC.StringTensorServer",
    ):
        return "str_list"

    if hasattr(shared_mem, "read_vec") and hasattr(shared_mem, "length") and hasattr(shared_mem, "get_raw_buffer"):
        return "str_list"

    basename = ""
    if hasattr(shared_mem, "getBasename"):
        basename = str(shared_mem.getBasename()).lower()

    if basename.endswith("names"):
        return "str_list"

    return "numeric"

class SharedDataBase(ABC):

    @abstractmethod
    def run(self):

        pass

    @abstractmethod
    def close(self):

        pass

    @abstractmethod
    def is_running(self):

        pass

    @abstractmethod
    def get_shared_mem(self):

        pass

    def get_shm_type(self):

        shared_mems = flatten_shared_mem(self.get_shared_mem())

        return [infer_shm_type(shared_mem) for shared_mem in shared_mems]

    def get_shm_sliceable(self):

        shm_types = self.get_shm_type()

        return [shm_type == "numeric" for shm_type in shm_types]
    
def is_shared_data_child(cls):

    return issubclass(cls,
             SharedDataBase)
